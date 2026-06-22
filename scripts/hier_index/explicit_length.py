from data.peptide import PeptideDataset
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmap_dir', type=str, required=True, help='Path to the mmap dataset directory')
    args = parser.parse_args()

    dataset = PeptideDataset(
        mmap_dir=args.mmap_dir
    )

    atom_lengths = []
    block_lengths = []
    with open(f'{args.mmap_dir}/test_write_permission.txt', 'w') as f:
        f.write('test')
    for i in tqdm(range(len(dataset))):
        props = dataset._properties[i]
        atom_len = props['pocket_num_atoms'] + props['ligand_num_atoms']
        block_len = props['pocket_num_blocks'] + props['ligand_num_blocks']
        atom_lengths.append(atom_len)
        block_lengths.append(block_len)
    # shape [N, 2]
    lengths = np.vstack([atom_lengths, block_lengths]).T
    np.save(f'{args.mmap_dir}/lengths.npy', lengths)