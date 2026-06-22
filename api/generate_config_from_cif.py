#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script to generate a config.yaml file from CIF files in a specified folder.
"""
import os
import argparse
import yaml


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate config.yaml from CIF files')
    parser.add_argument('--input_folder', type=str, required=True, help='Folder containing CIF files')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the generated config.yaml')
    parser.add_argument('--tgt_chain', type=str, default='R', help='Target chain identifier (default: R)')
    parser.add_argument('--lig_chain', type=str, default='L', help='Ligand chain identifier (default: L)')
    parser.add_argument('--pocket_dist_th', type=float, default=20.0, help='Pocket distance threshold (default: 20.0)')
    parser.add_argument('--n_samples', type=int, default=10, help='Number of samples per complex (default: 10)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size (default: 16)')
    args = parser.parse_args()

    # Collect all CIF files from the input folder
    cif_files = []
    for root, dirs, files in os.walk(args.input_folder):
        for file in files:
            if file.lower().endswith('.cif'):
                # Get relative path to maintain the structure similar to the demo config
                rel_path = os.path.join(root, file)
                cif_files.append(rel_path)

    if not cif_files:
        print(f"Warning: No CIF files found in {args.input_folder}")
        return

    # Create config structure
    config = {
        'dataset': {
            'class': 'FileDataset',
            'file_paths': cif_files,
            'tgt_chains': [args.tgt_chain] * len(cif_files),
            'lig_chains': [args.lig_chain] * len(cif_files),
            'pocket_dist_th': args.pocket_dist_th
        },
        'n_samples': args.n_samples,
        'batch_size': args.batch_size
    }

    # Ensure the save directory exists
    os.makedirs(args.save_dir, exist_ok=True)

    # Save the config to config.yaml
    config_path = os.path.join(args.save_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Config file has been generated at: {config_path}")
    print(f"Found {len(cif_files)} CIF files")


if __name__ == '__main__':
    main()