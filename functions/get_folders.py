# -*- coding: utf-8 -*-
"""
Created on Tue Dec 17 17:01:58 2024

@author: Hiroto Imamura
"""

import glob
import re
import os

import os
import re
import glob

def load_checkpoint_with_max_number(folder_path):
    # Find all .ckpt.index files in the folder
    ckpt_index_files = glob.glob(os.path.join(folder_path, '*.ckpt.index'))

    # Check if the list is empty
    if not ckpt_index_files:
        print("No .ckpt.index files found in the folder.")
        return None

    file_number_pairs = []
    for file in ckpt_index_files:
        # Extract the base name (e.g. "epoch_03173.ckpt.index")
        base_name = os.path.basename(file)
        # Remove the .ckpt.index part to extract the meaningful filename
        # e.g. "epoch_03173.ckpt.index" -> "epoch_03173"
        # We'll do this in two steps:
        # 1. Remove the ".index" extension
        base_name_no_index = base_name.replace('.ckpt.index', '.ckpt')
        # Now base_name_no_index might be "epoch_03173.ckpt"
        # Extract numbers from the filename
        numbers = re.findall(r'\d+', base_name_no_index)
        if numbers:
            # Assume the last number is the epoch number
            file_number_pairs.append((file, int(numbers[-1])))
        else:
            print(f"No numbers found in filename: {file}")

    if not file_number_pairs:
        print("No numbers found in any filenames.")
        return None

    # Get the file with the highest number (most recent epoch)
    max_file, max_number = max(file_number_pairs, key=lambda x: x[1])

    # Remove the .index to get the checkpoint prefix
    checkpoint_prefix = max_file.replace('.index', '')
    return checkpoint_prefix, max_number


def load_npy_with_max_number(folder_path):
    # Find all .npy files in the folder
    npy_files = glob.glob(os.path.join(folder_path, '*.npy'))

    # Check if the list is empty
    if not npy_files:
        print("No .npy files found in the folder.")
        return None

    file_number_pairs = []
    for file in npy_files:
        # Extract the base name (e.g., "data_03173.npy")
        base_name = os.path.basename(file)
        # Extract numbers from the filename
        numbers = re.findall(r'\d+', base_name)
        if numbers:
            # Assume the last number is the most relevant (e.g., epoch number)
            file_number_pairs.append((file, int(numbers[-1])))
        else:
            print(f"No numbers found in filename: {file}")

    if not file_number_pairs:
        print("No numbers found in any filenames.")
        return None

    # Get the file with the highest number
    max_file, max_number = max(file_number_pairs, key=lambda x: x[1])

    return max_file, max_number

def list_ckpt_checkpoints(folder_path):
    """
    Scan `folder_path` for '*.ckpt.index' files, extract epochs, and return:
      epochs: List[int]          sorted ascending
      paths:  List[str]          corresponding '.ckpt' prefixes
    """
    idx_paths = glob.glob(os.path.join(folder_path, '*.ckpt.index'))
    epoch_path_pairs = []
    for idx in idx_paths:
        name = os.path.basename(idx)                    # e.g. "epoch_00123.ckpt.index"
        prefix = idx[:-len('.index')]                   # ".../epoch_00123.ckpt"
        nums = re.findall(r'\d+', name)
        if not nums:
            continue
        epoch = int(nums[-1])
        epoch_path_pairs.append((epoch, prefix))

    # sort by epoch
    epoch_path_pairs.sort(key=lambda x: x[0])

    # unzip into two lists
    epochs, paths = zip(*epoch_path_pairs) if epoch_path_pairs else ([], [])
    return list(epochs), list(paths)
    
def load_most_recent_pt_file(folder_path):
    # Find all .pt files in the folder
    pt_files = glob.glob(os.path.join(folder_path, '*.ckpt'))
    
    # Check if the list is empty
    if not pt_files:
        print("No .pt files found in the folder.")
        return None
    
    # Get the most recent .pt file based on modification time
    most_recent_file = max(pt_files, key=os.path.getmtime)
    return most_recent_file