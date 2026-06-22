from data.peptide import PeptideDataset
import argparse
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mmap_dirs', type=str, nargs='+', required=True, help='Paths to the mmap dataset directories')
    parser.add_argument('--output_dir', type=str, required=True, help='Path to save the shuffled index.npy file')
    args = parser.parse_args()

    tot_len = 0

    for mmap_dir in args.mmap_dirs:
        dataset = PeptideDataset(
            mmap_dir=mmap_dir
        )
    
        tot_len += len(dataset)
    shuffled_indices = np.random.permutation(tot_len)
    np.save(f'{args.output_dir}/shuffled.npy', shuffled_indices)