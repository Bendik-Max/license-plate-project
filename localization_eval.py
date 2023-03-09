import LocalizationEvaluation
import argparse
from datetime import datetime


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample_freq', type=int, default=1)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()
    sample_freq = args.sample_freq
    print("Evaluation on training and testing datasets for categories 1-4.")
    print("This may take a while...")
    before = datetime.now()
    for i in range(1, 5):
        LocalizationEvaluation.evaluate_category_training(i, sample_freq)
        LocalizationEvaluation.evaluate_category_validation(i, sample_freq)
    after = datetime.now()
    print("Evaluation took: " + str((after - before).seconds) + " seconds.")
