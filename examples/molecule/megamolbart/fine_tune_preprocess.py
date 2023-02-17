import os
import pandas as pd
import argparse
from nemo.utils import logging


def _process_split(datafile, val_frac, test_frac, output_dir='/tmp/molecule_data/processed/', seed=0):
    logging.info(f'Splitting file {datafile} into train, validation, and test data')

    df = pd.read_csv(datafile, header=0)

    # Calculate sample sizes before size of dataframe changes
    test_samples = max(int(test_frac * df.shape[0]), 1)
    val_samples = max(int(val_frac * df.shape[0]), 1)

    test_df = df.sample(n=test_samples, random_state=seed)
    df = df.drop(test_df.index)  # remove test data from training data

    val_df = df.sample(n=val_samples, random_state=seed)
    df = df.drop(val_df.index)  # remove validation data from training data

    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val'), exist_ok=True)

    df.to_csv(f'{output_dir}/train/train.csv', index=False)
    test_df.to_csv(f'{output_dir}/test/test.csv', index=False)
    val_df.to_csv(f'{output_dir}/val/val.csv', index=False)

    del df
    del test_df
    del val_df

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="the csv data to be split", required=True)
    parser.add_argument("-vf", "--val-frac", type=float, help="Fraction of data to be validation data", required=True)
    parser.add_argument("-tf", "--test-frac", type=float, help="Fraction of data to be test data", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-o", "--output-dir", help="Output directory for split data", default="/tmp/molecule_data/processed/")
    args = parser.parse_args()

    _process_split(args.input, args.val_frac, args.test_frac, output_dir=args.output_dir, seed=args.seed)


if __name__ == '__main__':
    main()