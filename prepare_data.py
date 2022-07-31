import argparse

from src.processing.wa_converter import prepare_data


parser = argparse.ArgumentParser(description='CLI wrapper for NLP classifier using BERT')
parser.add_argument('--train-dir', default='train',
                    help='comma separated paths for the train datasets')
parser.add_argument('--test-dir', default='test',
                    help='path for the test dataset')
parser.add_argument('--datasets', default='headlines',
                    help='comma separated dataset names to prepare')


args = parser.parse_args()

train_dir = args.train_dir
test_dir = args.test_dir
datasets = args.datasets.split(',')


if __name__ == '__main__':
    prepare_data(train_dir, test_dir, datasets)
