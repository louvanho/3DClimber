import os

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ['WANDB_SILENT'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

# Patch the delete function on MLCloud to free up quota immediately
# import mlcloud_monkey_patch_truncation
# mlcloud_monkey_patch_truncation.patch_tf_delete()

# Must import cv2 early
# noinspection PyUnresolvedReferences
import cv2
#

# Must import TF early due to problem with PyCharm debugger otherwise.
# noinspection PyUnresolvedReferences
import tensorflow as tf
import argparse
import os.path as osp
import shlex
import socket
import sys

import matplotlib.pyplot as plt
import simplepyutils as spu
from simplepyutils import FLAGS, logger

from nlf.tf import tfu, util
from posepile.paths import DATA_ROOT


def initialize(args=None):
    spu.argparse.initialize_with_logfiles(
        get_parser(), logdir_root=f'{DATA_ROOT}/experiments', args=args)
    print(FLAGS)
    logger.info(f'-- Starting --')
    logger.info(f'Host: {socket.gethostname()}')
    logger.info(f'Process id (pid): {os.getpid()}')
    logger.info(f'Slurm job id: {os.environ.get("SLURM_JOB_ID")}')

    if FLAGS.comment:
        logger.info(f'Comment: {FLAGS.comment}')
    logger.info(f'Raw command: {" ".join(map(shlex.quote, sys.argv))}')
    logger.info(f'Parsed flags: {FLAGS}')
    tfu.set_data_format(FLAGS.data_format)
    tfu.set_dtype(dict(float32=tf.float32, float16=tf.float16, bfloat16=tf.bfloat16)[FLAGS.dtype])

    if FLAGS.batch_size_test is None:
        FLAGS.batch_size_test = FLAGS.batch_size

    if FLAGS.checkpoint_dir is None:
        FLAGS.checkpoint_dir = FLAGS.logdir

    FLAGS.checkpoint_dir = util.ensure_absolute_path(
        FLAGS.checkpoint_dir, root=f'{DATA_ROOT}/experiments')
    os.makedirs(FLAGS.checkpoint_dir, exist_ok=True)

    if not FLAGS.pred_path:
        FLAGS.pred_path = f'predictions_{FLAGS.dataset3d}.npz'
    base = osp.dirname(FLAGS.load_path) if FLAGS.load_path else FLAGS.checkpoint_dir
    FLAGS.pred_path = util.ensure_absolute_path(FLAGS.pred_path, base)

    if FLAGS.load_path:
        if FLAGS.load_path.endswith('.index') or FLAGS.load_path.endswith('.meta'):
            FLAGS.load_path = osp.splitext(FLAGS.load_path)[0]
        FLAGS.load_path = util.ensure_absolute_path(FLAGS.load_path, FLAGS.checkpoint_dir)

    for gpu in tf.config.experimental.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)

    tf.keras.utils.set_random_seed(FLAGS.seed)

    if FLAGS.viz:
        plt.switch_backend('TkAgg')

    FLAGS.backbone = FLAGS.backbone.replace('_', '-')

    if FLAGS.dtype in ('float16', 'bfloat16'):
        tf.keras.mixed_precision.set_global_policy(f'mixed_{FLAGS.dtype}')

    if FLAGS.stop_step is None:
        FLAGS.stop_step = FLAGS.training_steps


def get_parser():
    parser = argparse.ArgumentParser(
        description='Neural Localizer Fields for Continuous 3D Human Pose and Shape Estimation', allow_abbrev=False)

    parser.add_argument('--comment', type=str, default=None)
    parser.add_argument('--seed', type=int, default=1, help='Seed for the random number generators')
    parser.add_argument('--wandb-project', type=str, default='localizerfield')

    # Parallelism
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers to run. Default is min(12, num_cpus)')
    parser.add_argument('--multi-gpu', action=spu.argparse.BoolAction)

    # Task options (what to do)
    parser.add_argument('--train', action=spu.argparse.BoolAction, help='Train the model.')
    parser.add_argument('--predict', action=spu.argparse.BoolAction, help='Test the model.')
    parser.add_argument('--export-file', type=str, help='Export filename.')
    parser.add_argument('--pred-path', type=str, default=None)

    # Monitoring options
    parser.add_argument('--viz', action=spu.argparse.BoolAction,
                        help='Create graphical user interface for visualization.')

    # Loading and input processing options
    parser.add_argument('--load-path', type=str, default=None,
                        help='Path of model checkpoint to load in the beginning.')
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                        help='Directory path of model checkpoints.')
    parser.add_argument('--init-path', type=str, default=None,
                        help="""Path of the pretrained checkpoint to initialize from once
                        at the very start of training (i.e. not when resuming!).
                        To restore for resuming an existing training use the --load-path option.""")
    parser.add_argument('--load-backbone-from', type=str)

    # Augmentations, image preproc
    parser.add_argument('--proc-side', type=int, default=256,
                        help='Side length of image as processed by network.')

    parser.add_argument('--geom-aug', action=spu.argparse.BoolAction, default=True,
                        help='Training data augmentations such as rotation, scaling, translation '
                             'etc.')
    parser.add_argument('--hflip-aug', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--rot-aug', type=float,
                        help='Rotation augmentation in degrees.', default=20)
    parser.add_argument('--full-rot-aug-prob', type=float, default=0.1)
    parser.add_argument('--scale-aug-up', type=float,
                        help='Scale augmentation in percent.', default=25)
    parser.add_argument('--scale-aug-down', type=float,
                        help='Scale augmentation in percent.', default=25)
    parser.add_argument('--shift-aug', type=float,
                        help='Shift augmentation in percent.', default=10)

    parser.add_argument('--occlude-aug-prob', type=float, default=0.5)
    parser.add_argument('--occlude-aug-prob-2d', type=float, default=0.7)
    parser.add_argument('--occlude-aug-scale', type=float, default=1)
    parser.add_argument('--background-aug-prob', type=float, default=0.7)
    parser.add_argument('--color-aug', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--partial-visibility-prob', type=float, default=0.15)
    parser.add_argument('--jpeg-aug-prob', type=float, default=0.25)
    parser.add_argument('--augment-border', action=spu.argparse.BoolAction)
    parser.add_argument('--border-value', type=int, default=0)

    parser.add_argument('--test-aug', action=spu.argparse.BoolAction,
                        help='Apply augmentations to test images.')
    parser.add_argument('--test-time-mirror-aug', action=spu.argparse.BoolAction)

    parser.add_argument('--antialias-train', type=int, default=1)
    parser.add_argument('--antialias-test', type=int, default=1)  # 4 can be more accurate
    parser.add_argument(
        '--image-interpolation-train', type=str, default='linear')  # 'nearest' can be faster
    parser.add_argument('--image-interpolation-test', type=str, default='linear')

    # Data choices for Human3.6M
    parser.add_argument('--test-subjects', type=str, default=None, help='Test subjects.')
    parser.add_argument('--valid-subjects', type=str, default=None, help='Validation subjects.')
    parser.add_argument('--train-subjects', type=str, default=None, help='Training subjects.')

    # What to train, validate and test on
    parser.add_argument('--train-on', type=str, default='train', help='Training part.')
    parser.add_argument('--validate-on', type=str, default='valid', help='Validation part.')
    parser.add_argument('--test-on', type=str, default='test', help='Test part.')

    # Training options
    parser.add_argument('--dtype', type=str, default='float16',
                        choices=['float16', 'float32', 'bfloat16'],
                        help='The floating point type to use for computations.')
    parser.add_argument('--data-format', type=str, default='NHWC',
                        choices=['NHWC', 'NCHW'], help='Data format used internally.')

    parser.add_argument('--validate-period', type=int, default=None,
                        help='Periodically validate during training, every this many steps.')
    parser.add_argument('--checkpoint-period', type=int, default=2000)

    # Optimizer
    parser.add_argument('--training-steps', type=int)
    parser.add_argument('--completed-steps', type=int)
    parser.add_argument('--stop-step', type=int, default=None)

    parser.add_argument('--weight-decay', type=float, default=3e-3)
    parser.add_argument('--decay-field', action=spu.argparse.BoolAction)

    parser.add_argument('--base-learning-rate', type=float, default=2.121e-4,
                        help='Learning rate of the optimizer.')
    parser.add_argument('--dual-finetune-lr', action=spu.argparse.BoolAction)
    parser.add_argument('--lr-cooldown-fraction', type=float, default=0.08)
    parser.add_argument('--backbone-lr-factor', type=float, default=0.15)
    parser.add_argument('--field-lr-factor', type=float, default=1)

    parser.add_argument('--loss-scale', type=float, default=128)
    parser.add_argument('--dynamic-loss-scale', action=spu.argparse.BoolAction)
    parser.add_argument('--ema-momentum', type=float, default=1,
                        help='The momentum of the exponential moving average in Polyak averaging.')
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--force-grad-accum', action=spu.argparse.BoolAction,
                        help='Use a GradientAccumulationOptimizer even when grad-accum-steps=1.'
                             'Useful for being able to switch gradient accumulation on and off'
                             'and have the checkpoints still work.')
    parser.add_argument('--finetune-in-inference-mode', type=int, default=0)
    parser.add_argument('--constrain-kernel-norm', type=float, default=20)
    parser.add_argument('--constraint-rate', type=float, default=1)

    # Batching
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--batch-size-test', type=int, default=150)
    parser.add_argument('--batch-size-parametric', type=int, default=64)
    parser.add_argument('--batch-size-densepose', type=int, default=20)
    parser.add_argument('--batch-size-2d', type=int, default=32)


    parser.add_argument('--stride-train', type=int, default=32)
    parser.add_argument('--stride-test', type=int, default=32)
    parser.add_argument('--centered-stride', action=spu.argparse.BoolAction, default=True)

    # Data
    parser.add_argument('--image-barecat-path', type=str)
    parser.add_argument('--dataset3d', type=str, default='h36m')
    parser.add_argument('--dataset3d-kp', type=str)
    parser.add_argument('--dataset2d', type=str, default='mpii')
    parser.add_argument('--dataset-dense', type=str)

    # Model
    parser.add_argument('--backbone', type=str, default='efficientnetv2-s',
                        help='Backbone of the predictor network.')
    parser.add_argument('--depth', type=int, default=8,
                        help='Number of voxels along the z axis for volumetric prediction')
    parser.add_argument('--box-size-m', type=float, default=2.2)

    parser.add_argument('--weak-perspective', action=spu.argparse.BoolAction)
    parser.add_argument('--mix-3d-inside-fov', type=float, default=0.5)
    parser.add_argument('--fullpersp-l2-regul', type=float, default=1e-4)
    parser.add_argument('--depth-on-2d-heatmap', action=spu.argparse.BoolAction, default=True)

    # Batchnorm
    parser.add_argument('--ghost-bn', type=str, default='')
    parser.add_argument('--batch-renorm', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--renorm-limit-scale', type=float, default=1)


    # Loss
    parser.add_argument('--nll-loss', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--beta-nll', type=float, default=1)
    parser.add_argument('--charb-eps', type=float, default=0)
    parser.add_argument('--uncert-bias', type=float, default=0.0)
    parser.add_argument('--uncert-bias2', type=float, default=0.001)

    parser.add_argument('--loss-factor-param', type=float, default=1.0)
    parser.add_argument('--loss-factor-kp', type=float, default=0.75)
    parser.add_argument('--loss-factor-dense', type=float, default=0.05)
    parser.add_argument('--loss-factor-2d', type=float, default=0.1)

    parser.add_argument('--absloss-factor', type=float, default=0.1)
    parser.add_argument('--absloss-start-step', type=int, default=5000)

    parser.add_argument('--mean-relative', action=spu.argparse.BoolAction, default=True)

    # Field
    parser.add_argument('--backbone-linking-layer', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--backbone-link-dim', type=int, default=512)
    parser.add_argument('--add-neck', action=spu.argparse.BoolAction, default=True)

    parser.add_argument('--field-type', type=str, default='gps')
    parser.add_argument('--field-posenc-dim', type=int, default=1024)
    parser.add_argument('--field-hidden-layers', type=int, default=1)
    parser.add_argument('--field-hidden-size', type=int, default=384)

    parser.add_argument('--lbo-concat-coord', action=spu.argparse.BoolAction)
    parser.add_argument('--lbo-eigval-regul', type=float, default=1e-4)
    parser.add_argument('--lbo-zero-first', action=spu.argparse.BoolAction)
    parser.add_argument('--init-lbo', action=spu.argparse.BoolAction, default=True)
    parser.add_argument('--finetune-gps', action=spu.argparse.BoolAction, default=True)

    # Canonical point sampling
    parser.add_argument('--num-points', type=int, default=1024)
    parser.add_argument('--num-surface-points', type=int, default=640)
    parser.add_argument('--num-internal-points', type=int, default=384)
    parser.add_argument('--trainable-canonical-joints', action=spu.argparse.BoolAction,
                        default=True)

    parser.add_argument('--more-onlyskel', action=spu.argparse.BoolAction)

    parser.add_argument('--fix-uncert-factor', action=spu.argparse.BoolAction)

    parser.add_argument('--custom', type=str, default='')
    return parser
