import argparse

import einops
import numpy as np
import torch

torch  # this is a dummy line to make pycharm respect the order
import tensorflow as tf

from simplepyutils import logger
import nlf.pt.backbones.efficientnet as effnet_pytorch
import nlf.pt.models.nlf_model as pt_nlf_model
import nlf.pt.models.field as pt_field
import nlf.tf.backbones.builder as tf_backbones_builder
from nlf.tf import init as tf_init, tfu
import nlf.tf.model.field as tf_field
from nlf.pt.util import get_config

FLAGS = argparse.Namespace()


def initialize():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-model-path', type=str)
    parser.add_argument('--output-model-path', type=str)
    parser.add_argument('--config-name', type=str, default='convert_from_tf_s')
    parser.parse_args(namespace=FLAGS)
    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)


def main():
    torch.set_printoptions(precision=10)
    initialize()
    logger.info('Creating PyTorch model...')
    with torch.device('cuda'), torch.amp.autocast('cuda'):
        model_pytorch = create_pytorch_model()
    logger.info('Loading TF model...')
    model_tf = load_tensorflow_model()
    logger.info('Copying weights...')
    copy_efficientnetv2_weights(model_tf=model_tf, model_pytorch=model_pytorch)
    copy_field_weights(
        model_tf=model_tf.heatmap_head.weight_field,
        model_pytorch=model_pytorch.heatmap_head.weight_field)

    logger.info('Saving PyTorch model...')
    torch.save(model_pytorch.state_dict(), FLAGS.output_model_path)


def load_tensorflow_model():
    cfg = get_config(FLAGS.config_name)
    tf_init.initialize(
        f'--backbone=efficientnetv2-{cfg.efficientnet_size} '
        f'--proc-side={cfg.proc_side}')
    backbone, normalizer = tf_backbones_builder.build_backbone()
    weight_field = tf_field.build_field()

    import nlf.tf.model.nlf_model as tf_nlf_model
    model = tf_nlf_model.NLFModel(backbone, weight_field, normalizer)
    inp = tf.keras.Input(shape=(None, None, 3), dtype=tfu.get_dtype())
    intr = tf.keras.Input(shape=(3, 3), dtype=tf.float32)
    canonical_loc = tf.keras.Input(shape=(None, 3), dtype=tf.float32, ragged=True)
    model((inp, intr, canonical_loc), training=False)
    ckpt = tf.train.Checkpoint(model=model)
    s = ckpt.restore(FLAGS.input_model_path)
    s.expect_partial()
    return model


def create_pytorch_model():
    cfg = get_config(FLAGS.config_name)
    backbone_raw = getattr(effnet_pytorch, f'efficientnet_v2_{cfg.efficientnet_size}')()
    preproc_layer = effnet_pytorch.PreprocLayer()
    backbone = torch.nn.Sequential(preproc_layer, backbone_raw.features)
    weight_field = pt_field.build_field()
    model = pt_nlf_model.NLFModel(backbone, weight_field)
    model.eval()

    input_image = torch.zeros((1, 3, cfg.proc_side, cfg.proc_side), dtype=torch.float32)
    intrinsics = torch.eye(3, dtype=torch.float32)[np.newaxis]
    canonical_points = torch.tensor(np.random.randn(16, 3), dtype=torch.float32)
    model.predict_multi_same_canonicals(input_image, intrinsics, canonical_points)
    return model


def rearrange_tf_to_pt(value, depthwise=False):
    if value.ndim == 4:
        if depthwise:
            return einops.rearrange(value, 'h w c_in c_out -> c_in c_out h w')
        else:
            return einops.rearrange(value, 'h w c_in c_out -> c_out c_in h w')
    elif value.ndim == 2:
        return einops.rearrange(value, 'c_in c_out -> c_out c_in')
    elif value.ndim == 1:
        return value


def copy_efficientnetv2_weights(model_tf, model_pytorch):
    cfg = get_config(FLAGS.config_name)
    if cfg.efficientnet_size == 'l':
        block_counts = [[4], [7, 7], [10, 19, 25, 7]]
    elif cfg.efficientnet_size == 'm':
        block_counts = [[3], [5, 5], [7, 14, 18, 5]]
    elif cfg.efficientnet_size == 's':
        block_counts = [[2], [4, 4], [6, 9, 15]]
    else:
        raise ValueError(f'Unknown efficientnet size: {cfg.efficientnet_size}')

    weights_tf = {v.name: v.numpy() for v in model_tf.weights}
    weights_pt = model_pytorch.state_dict()

    pairs = []
    i_block_tf = 0
    i_block_pt = 0
    effnet_name = f'efficientnetv2-{cfg.efficientnet_size}'

    def increment_tf_block_counter():
        nonlocal i_block_tf
        i_block_tf += 1

    def increment_pt_block_counter():
        nonlocal i_block_pt
        i_block_pt += 1

    def add(a, b):
        pairs.append((a, b))

    def add_batchnorm(name_pt, name_tf):
        add(f'{name_pt}.weight', f'{name_tf}/gamma:0')
        add(f'{name_pt}.bias', f'{name_tf}/beta:0')

        add(f'{name_pt}.running_mean', f'{name_tf}/moving_mean:0')
        add(f'{name_pt}.running_var', f'{name_tf}/moving_variance:0')

    def add_stem(name_pt, name_tf):
        add(f'{name_pt}.0.weight', f'{name_tf}/conv2d/kernel:0')
        add_batchnorm(f'{name_pt}.1', f'{name_tf}/tpu_batch_normalization')

    def add_simplest_block(name_pt, name_tf):
        add(f'{name_pt}.block.0.0.weight', f'{name_tf}/conv2d/kernel:0')
        add_batchnorm(f'{name_pt}.block.0.1', f'{name_tf}/tpu_batch_normalization')
        increment_tf_block_counter()

    def add_simple_block(name_pt, name_tf):
        add(f'{name_pt}.block.0.0.weight', f'{name_tf}/conv2d/kernel:0')
        add_batchnorm(f'{name_pt}.block.0.1', f'{name_tf}/tpu_batch_normalization')

        add(f'{name_pt}.block.1.0.weight', f'{name_tf}/conv2d_1/kernel:0')
        add_batchnorm(f'{name_pt}.block.1.1', f'{name_tf}/tpu_batch_normalization_1')
        increment_tf_block_counter()

    def add_depthwise_block(name_pt, name_tf):
        add(f'{name_pt}.block.0.0.weight', f'{name_tf}/conv2d/kernel:0')
        add_batchnorm(f'{name_pt}.block.0.1', f'{name_tf}/tpu_batch_normalization')

        add(f'{name_pt}.block.1.0.weight', f'{name_tf}/depthwise_conv2d/depthwise_kernel:0')
        add_batchnorm(f'{name_pt}.block.1.1', f'{name_tf}/tpu_batch_normalization_1')

        add(f'{name_pt}.block.2.fc1.weight', f'{name_tf}/se/conv2d/kernel:0')
        add(f'{name_pt}.block.2.fc1.bias', f'{name_tf}/se/conv2d/bias:0')
        add(f'{name_pt}.block.2.fc2.weight', f'{name_tf}/se/conv2d_1/kernel:0')
        add(f'{name_pt}.block.2.fc2.bias', f'{name_tf}/se/conv2d_1/bias:0')

        add(f'{name_pt}.block.3.0.weight', f'{name_tf}/conv2d_1/kernel:0')
        add_batchnorm(f'{name_pt}.block.3.1', f'{name_tf}/tpu_batch_normalization_2')
        increment_tf_block_counter()

    def add_penultimate(name_pt, name_tf):
        add(f'{name_pt}.0.weight', f'{name_tf}/conv2d/kernel:0')
        add_batchnorm(f'{name_pt}.1', f'{name_tf}/tpu_batch_normalization')

    def add_blocks(n_blockss, block_function):
        for n_blocks in n_blockss:
            for i_subblock_pt in range(n_blocks):
                block_function(f'backbone.1.{i_block_pt}.{i_subblock_pt}',
                               f'{effnet_name}/blocks_{i_block_tf}')
            increment_pt_block_counter()

    add_stem(f'backbone.1.{i_block_pt}', f'{effnet_name}/stem')
    increment_pt_block_counter()

    add_blocks(block_counts[0], add_simplest_block)
    add_blocks(block_counts[1], add_simple_block)
    add_blocks(block_counts[2], add_depthwise_block)

    add_penultimate(f'backbone.1.{i_block_pt}', f'{effnet_name}/head')

    add('heatmap_head.layer.0.weight', 'conv2d/kernel:0')
    add_batchnorm('heatmap_head.layer.1', 'tpu_batch_normalization')

    for name_pt, name_tf in pairs:
        weights_pt[name_pt][:] = torch.from_numpy(
            rearrange_tf_to_pt(weights_tf[name_tf], depthwise='depthwise' in name_tf))

    weights_pt['canonical_lefts'][:] = torch.from_numpy(weights_tf['canonical_locs_left:0'])
    weights_pt['canonical_centers'][:] = torch.from_numpy(weights_tf['canonical_locs_centers:0'])

    model_pytorch.load_state_dict(weights_pt)


def copy_field_weights(model_tf, model_pytorch):
    weights_tf = [v.numpy() for v in model_tf.weights]
    weights_pt = model_pytorch.state_dict()

    for name_pt, weight_tf in zip(weights_pt, weights_tf):
        weights_pt[name_pt][:] = torch.from_numpy(rearrange_tf_to_pt(weight_tf))

    model_pytorch.load_state_dict(weights_pt)


if __name__ == '__main__':
    main()
