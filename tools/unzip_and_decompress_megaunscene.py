#!/usr/bin/env python3
"""
Unzip .zip files and decompress depth maps in-place from zipped MegaUnScene release download.
"""

import argparse
import os
import numpy as np
from numcodecs import Blosc
import zipfile


def decompress_depth_map(src_path, dst_path):
    """
    Decompress a depth map file that was compressed with Blosc.
    
    Args:
        src_path: Source compressed depth map file path (.compressed)
        dst_path: Destination decompressed .npy file path
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read compressed data with shape header
        with open(src_path, 'rb') as f:
            # Read shape (2 uint32s = 8 bytes)
            shape_bytes = f.read(8)
            shape = tuple(np.frombuffer(shape_bytes, dtype=np.uint32))
            
            # Read compressed data
            depth_compressed = f.read()
        
        # Decompress with Blosc
        compressor = Blosc(cname='zstd', clevel=5, shuffle=2)
        raw_bytes = compressor.decode(depth_compressed)
        
        # Convert bytes back to array and reshape
        depth_data = np.frombuffer(raw_bytes, dtype=np.float32).reshape(shape)
        
        assert depth_data.dtype == np.float32, f"Expected float32, got {depth_data.dtype}"
        np.save(dst_path, depth_data)
        
        return True
    except Exception as e:
        print(f"    ERROR decompressing {src_path}: {e}")
        return False


def unzip_zips_inplace(recon_dir):
    """
    Unzip all .zip files in-place.
    
    Args:
        recon_dir: Reconstruction directory containing .zip files
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Unzip images.zip in-place
        images_zip = os.path.join(recon_dir, "images.zip")
        if os.path.exists(images_zip):
            images_dir = os.path.join(recon_dir, "images")
            os.makedirs(images_dir, exist_ok=True)
            with zipfile.ZipFile(images_zip, 'r') as zf:
                zf.extractall(images_dir)
            os.remove(images_zip)
        
        # Unzip depth_maps.zip in-place
        depth_maps_zip = os.path.join(recon_dir, "depth_maps.zip")
        if os.path.exists(depth_maps_zip):
            depth_maps_dir = os.path.join(recon_dir, "depth_maps")
            os.makedirs(depth_maps_dir, exist_ok=True)
            with zipfile.ZipFile(depth_maps_zip, 'r') as zf:
                zf.extractall(depth_maps_dir)
            os.remove(depth_maps_zip)
        
        return True
    except Exception as e:
        print(f"    ERROR unzipping {recon_dir}: {e}")
        return False


def decompress_compressed_files_inplace(recon_dir):
    """
    Decompress all .compressed files in the depth_maps folder.
    
    Args:
        recon_dir: Reconstruction directory containing depth_maps folder
    
    Returns:
        True if successful, False otherwise
    """
    try:
        depth_maps_dir = os.path.join(recon_dir, "depth_maps")
        if not os.path.exists(depth_maps_dir):
            return True
        
        for root, dirs, files in os.walk(depth_maps_dir):
            for file in files:
                if file.endswith('.compressed'):
                    src_file = os.path.join(root, file)
                    dst_file = src_file[:-len('.compressed')]
                    decompress_depth_map(src_file, dst_file)
        
        return True
    except Exception as e:
        print(f"    ERROR decompressing {recon_dir}: {e}")
        return False


def unzip_megaunscene_release_inplace(megaunscene_base):
    """
    Unzip and decompress all scenes in MegaUnScene directory in-place.
    
    Args:
        megaunscene_base: Base path to MegaUnScene directory
    """
    scenes_dir = os.path.join(megaunscene_base, "scenes")
    
    if not os.path.exists(scenes_dir):
        print(f"ERROR: scenes directory not found at {scenes_dir}")
        return False
    
    # Collect all scene/recon_id pairs
    scene_recon_pairs = []
    for scene in os.listdir(scenes_dir):
        scene_path = os.path.join(scenes_dir, scene)
        if not os.path.isdir(scene_path):
            continue
        
        for recon_id_str in os.listdir(scene_path):
            recon_path = os.path.join(scene_path, recon_id_str)
            if os.path.isdir(recon_path):
                try:
                    recon_id = int(recon_id_str)
                    scene_recon_pairs.append((scene, recon_id))
                except ValueError:
                    continue
    
    print(f"Found {len(scene_recon_pairs)} scene/recon pairs to unzip and decompress in-place")
    
    # Pass 1: Unzip all .zip files
    print("\n=== PASS 1: Unzipping .zip files ===")
    unzip_success = 0
    for i, (scene, recon_id) in enumerate(scene_recon_pairs):
        recon_path = os.path.join(scenes_dir, scene, str(recon_id))
        
        if not os.path.exists(recon_path):
            print(f"[{i+1}/{len(scene_recon_pairs)}] SKIP {scene}/{recon_id} - path not found")
            continue
        
        print(f"[{i+1}/{len(scene_recon_pairs)}] Unzipping {scene}/{recon_id}...")
        
        if unzip_zips_inplace(recon_path):
            unzip_success += 1
            print("  Successfully unzipped")
        else:
            print(f"  Failed to unzip {scene}/{recon_id}")
    
    print(f"\nUnzipped {unzip_success}/{len(scene_recon_pairs)} scene/recon pairs")
    
    # Pass 2: Decompress all .compressed files
    print("\n=== PASS 2: Decompressing .compressed files ===")
    decompress_success = 0
    for i, (scene, recon_id) in enumerate(scene_recon_pairs):
        recon_path = os.path.join(scenes_dir, scene, str(recon_id))
        
        if not os.path.exists(recon_path):
            print(f"[{i+1}/{len(scene_recon_pairs)}] SKIP {scene}/{recon_id} - path not found")
            continue
        
        print(f"[{i+1}/{len(scene_recon_pairs)}] Decompressing {scene}/{recon_id}...")
        
        if decompress_compressed_files_inplace(recon_path):
            decompress_success += 1
            print("  Successfully decompressed")
        else:
            print(f"  Failed to decompress {scene}/{recon_id}")
    
    print(f"\nDecompressed {decompress_success}/{len(scene_recon_pairs)} scene/recon pairs")
    print(f"\nSuccessfully completed {unzip_success}/{len(scene_recon_pairs)} unzip and {decompress_success}/{len(scene_recon_pairs)} decompress")
    return unzip_success == len(scene_recon_pairs) and decompress_success == len(scene_recon_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unzip and decompress depth maps in-place from MegaUnScene releases")
    parser.add_argument("--megaunscene_base", type=str,
                        help="Base path to MegaUnScene directory")
    args = parser.parse_args()
    
    unzip_megaunscene_release_inplace(args.megaunscene_base)
