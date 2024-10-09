import argparse
import numpy as np
import pandas as pd


def f(row):
    res = []
    for i in range(len(row)):
        if row[i] == 1:
            res.append(i)
    return res


def main(example_path: str, probs_path: str, probs_new_path: str, output_csv: str):
    sample = pd.read_csv(example_path)
    probs = np.load(probs_path)
    probs_new = np.load(probs_new_path)

    tholds = []
    for thold_prep in range(1000, 5501, 125):
        thold = thold_prep / 10000
        tholds.append(thold)

        pred = (probs > thold).astype(int)
        sample[f'target_{thold}'] = [f(row) for row in pred]
        sample[f'target_{thold}'] = sample[f'target_{thold}'].apply(lambda x: ' '.join(map(str, x)))

        pred_new = (probs_new > thold).astype(int)
        sample[f'target_new_{thold}'] = [f(row) for row in pred_new]
        sample[f'target_new_{thold}'] = sample[f'target_new_{thold}'].apply(lambda x: ' '.join(map(str, x)))

    for thold in tholds:
        sample[f'target_{thold}'] = sample[f'target_{thold}'].replace('', np.nan)
        sample[f'target_new_{thold}'] = sample[f'target_new_{thold}'].replace('', np.nan)

    for thold in reversed(tholds):
        print(f'curr thold: {thold}')
        print(sample[f'target_{thold}'].isna().sum())
        sample[f'target_{thold}'] = sample[f'target_{thold}'].fillna(sample[f'target_new_{thold}'])
        print(sample[f'target_{thold}'].isna().sum())
        print()

    for thold in reversed(tholds):
        print(f'curr thold: {thold}')
        print(sample['target_0.55'].isna().sum())
        sample['target_0.55'] = sample['target_0.55'].fillna(sample[f'target_{thold}'])
        print(sample['target_0.55'].isna().sum())
        print()

    sample['target_0.55'] = sample['target_0.55'].fillna(39)
    
    def keep_only_19_39(row):
        labels = row.split()
        if '19' in labels and len(labels) > 1:
            return '19'
        elif '39' in labels and len(labels) > 1:
            return '39'
        return row

    sample['target_0.55'] = sample['target_0.55'].apply(keep_only_19_39)

    preds = pd.DataFrame({'index': sample['index'], 'target': sample['target_0.55']})
    preds.to_csv(output_csv, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert predictions from numpy arrays to CSV.')
    parser.add_argument('--example', type=str, required=True, help='Path to the example CSV file.')
    parser.add_argument('--probs', type=str, required=True, help='Path to the first numpy array.')
    parser.add_argument('--probs_new', type=str, required=True, help='Path to the second numpy array.')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file name.')

    args = parser.parse_args()
    main(args.example, args.probs, args.probs_new, args.output)
