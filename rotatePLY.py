#!/usr/bin/env python3
import numpy as np
import argparse
import os
import re
from plyfile import PlyData, PlyElement

def read_images_txt(images_txt_path):
    """
    Read the images.txt file and find the line ending with ref.jpg
    Returns the quaternion and translation parameters
    """
    with open(images_txt_path, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines starting with '#'
    data_lines = [line for line in lines if not line.startswith('#')]
    
    # Process pairs of lines (each image has 2 lines)
    for i in range(0, len(data_lines), 2):
        if i + 1 < len(data_lines):  # Ensure we have a pair
            line = data_lines[i].strip()
            parts = line.split()
            
            # Check if this line contains an image name ending with ref.jpg
            if parts[-1].lower().endswith('ref.jpg'):
                # Extract quaternion (QW, QX, QY, QZ) and translation (TX, TY, TZ)
                qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
                
                print(f"Found reference image: {parts[-1]}")
                print(f"Quaternion: [{qw}, {qx}, {qy}, {qz}]")
                print(f"Translation: [{tx}, {ty}, {tz}]")
                
                return qw, qx, qy, qz, tx, ty, tz
    
    raise ValueError("No line ending with 'ref.jpg' found in the images.txt file")

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Convert quaternion (Hamilton convention) to rotation matrix
    """
    # Normalize quaternion
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw /= norm
    qx /= norm
    qy /= norm
    qz /= norm
    
    # Build rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    
    return R

def scale_pointcloud(vertices, scale_factor, camera_position=None):
    """
    Scale the point cloud by the given factor relative to a reference point.
    If camera_position is not provided, origin (0,0,0) is used as reference.
    """
    # Extract just the xyz coordinates
    xyz = vertices[:, :3].copy()
    
    # If camera position is not provided, use origin as reference
    if camera_position is None:
        camera_position = np.zeros(3)
    
    # Scale points: p_new = camera_pos + (p - camera_pos) / scale_factor
    # This moves points closer to the camera when scale_factor > 1
    scaled_xyz = camera_position + (xyz - camera_position) / scale_factor
    
    # Replace the xyz coordinates in the vertices array
    scaled_vertices = vertices.copy()
    scaled_vertices[:, :3] = scaled_xyz
    
    return scaled_vertices

def sample_pointcloud(vertices, sample_rate):
    """
    Sample the point cloud by keeping only every n-th point.
    
    Args:
        vertices: The point cloud vertices array
        sample_rate: Keep 1 out of every sample_rate points
        
    Returns:
        Sampled vertices array
    """
    if sample_rate <= 1:
        return vertices
    
    # Use numpy's fancy indexing to select every sample_rate-th point
    indices = np.arange(0, len(vertices), sample_rate)
    sampled_vertices = vertices[indices]
    
    print(f"Sampled point cloud from {len(vertices)} to {len(sampled_vertices)} points")
    return sampled_vertices

def inverse_transform(R, t):
    """
    Compute the inverse transform of R and t:
    - R_inv = R.T
    - t_inv = -R_inv @ t
    """
    # Inverse rotation is the transpose of the rotation matrix
    R_inv = R.T
    
    # Inverse translation: move origin to t and apply inverse rotation
    t_inv = -R_inv @ t
    
    return R_inv, t_inv

def read_ply_file(ply_path):
    """
    Read PLY file using plyfile library which handles both ASCII and binary formats
    """
    try:
        plydata = PlyData.read(ply_path)
        vertices = np.vstack([
            plydata['vertex']['x'],
            plydata['vertex']['y'],
            plydata['vertex']['z']
        ]).T
        
        # Get all vertex properties (coordinates and any additional properties like color)
        vertex_data = []
        vertex_properties = []
        
        # Get all property names
        for prop in plydata['vertex'].properties:
            vertex_properties.append(prop.name)
        
        # Extract all vertex data including properties like color
        for i in range(len(plydata['vertex'])):
            vertex = []
            for prop in vertex_properties:
                vertex.append(plydata['vertex'][prop][i])
            vertex_data.append(vertex)
        
        # Extract faces if they exist
        faces = []
        if 'face' in plydata:
            for i in range(len(plydata['face'])):
                faces.append(plydata['face'][i][0])
        
        return plydata, np.array(vertex_data), faces, vertex_properties
        
    except Exception as e:
        print(f"Error reading PLY file: {e}")
        raise

def transform_ply(vertices, R, t):
    """
    Apply rotation and translation to vertices
    """
    # Extract just the xyz coordinates
    xyz = vertices[:, :3]
    
    # Apply transformation: R*p + t
    transformed_xyz = np.dot(xyz, R.T) + t
    
    # Replace the xyz coordinates in the vertices array
    transformed_vertices = vertices.copy()
    transformed_vertices[:, :3] = transformed_xyz
    
    return transformed_vertices

def write_ply_file(output_path, plydata, transformed_vertices, vertex_properties):
    """
    Write transformed PLY file
    """
    # Create a new structured array for the vertices
    vertex_data = []
    for i in range(transformed_vertices.shape[0]):
        vertex = []
        for j, prop in enumerate(vertex_properties):
            if j < 3:  # x, y, z coordinates
                vertex.append(transformed_vertices[i, j])
            else:  # Other properties like color
                vertex.append(transformed_vertices[i, j])
        vertex_data.append(tuple(vertex))
    
    # Create the vertex element with the transformed data
    vertex_dtype = [(name, plydata['vertex'][name].dtype) for name in vertex_properties]
    vertices = np.array(vertex_data, dtype=vertex_dtype)
    el = PlyElement.describe(vertices, 'vertex')
    
    # Create elements list with the vertex element 
    elements = [el]
    
    # If we've sampled the point cloud, we need to remove the faces
    # since the indices will no longer match
    if len(transformed_vertices) != len(plydata['vertex']):
        print("Point cloud was sampled - removing faces from output PLY")
    else:
        # Only include faces if the vertex count hasn't changed
        if 'face' in plydata:
            elements.append(plydata['face'])
    
    # Create and write the new PLY data
    PlyData(elements, text=plydata.text).write(output_path)

def main():
    parser = argparse.ArgumentParser(description='Transform a PLY file using quaternion and translation from images.txt')
    parser.add_argument('--images_txt', type=str, required=True, help='Path to the images.txt file')
    parser.add_argument('--ply_file', type=str, required=True, help='Path to the PLY file to be transformed')
    parser.add_argument('--output', type=str, help='Path to save the transformed PLY file (default: input_transformed.ply)')
    parser.add_argument('--inverse', action='store_true', help='Apply inverse transformation instead of original')
    parser.add_argument('--scale', type=float, default=4, help='Scale factor to apply to the point cloud')
    parser.add_argument('--sample_rate', type=int, default=1, help='Sample the point cloud by keeping 1 out of every N points')
    
    args = parser.parse_args()
    
    # If output is not specified, use input filename with _transformed suffix
    if not args.output:
        base_name = os.path.splitext(args.ply_file)[0]
        args.output = f"{base_name}_transformed.ply"
    
    # Read transformation parameters from images.txt
    qw, qx, qy, qz, tx, ty, tz = read_images_txt(args.images_txt)
    print(f"Found transformation parameters:")
    print(f"Quaternion: [{qw}, {qx}, {qy}, {qz}]")
    print(f"Translation: [{tx}, {ty}, {tz}]")
    
    # Convert quaternion to rotation matrix
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    t = np.array([tx, ty, tz])
    
    # Read PLY file
    plydata, vertices, faces, vertex_properties = read_ply_file(args.ply_file)
    print(f"Read PLY file with {len(vertices)} vertices and {len(faces)} faces")
    
    # Step 1: Sample the point cloud if requested
    if args.sample_rate > 1:
        print(f"1. Sampling point cloud with rate 1/{args.sample_rate}")
        vertices = sample_pointcloud(vertices, args.sample_rate)
    
    # Step 2: Apply transformation from images.txt
    print(f"{'2' if args.sample_rate > 1 else '1'}. Applying transformation from images.txt")
    transformed_vertices = transform_ply(vertices, R, t)
    
    # Step 3: Scale the point cloud
    scale_factor = args.scale
    if scale_factor != 1.0:
        step_num = 3 if args.sample_rate > 1 else 2
        print(f"{step_num}. Scaling point cloud by factor: {scale_factor}")
        
        # Camera is at origin in camera coordinate system
        camera_position = np.zeros(3)
        
        # Apply scaling (bringing points closer to camera)
        final_vertices = scale_pointcloud(transformed_vertices, scale_factor, camera_position)
        
        print(f"Point cloud scaled down by factor of {scale_factor}")
    else:
        final_vertices = transformed_vertices
    
    # Write transformed and scaled PLY
    write_ply_file(args.output, plydata, final_vertices, vertex_properties)
    
    print(f"Transformed and scaled PLY saved to: {args.output}")
    if args.sample_rate > 1:
        print(f"Point cloud was sampled down to {100/args.sample_rate:.1f}% of original size")

if __name__ == "__main__":
    main()