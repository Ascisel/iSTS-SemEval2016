import argparse

import torch
import warnings

from src.operations.evaluation import evaluate_separate_model, evaluate_connected_model
from src.processing.wa_converter import prepare_test_wa_file


parser = argparse.ArgumentParser(description='CLI wrapper for NLP classifier using BERT')
parser.add_argument('--train-paths', default='train/processed_headlines.xml',
                    help='comma separated paths for the train datasets')
parser.add_argument('--test-path', default='test/processed_headlines.xml',
                    help='path for the test dataset')
parser.add_argument('--model', default='connected',
                    help='which model will be trained and evaluated')
parser.add_argument('--classifier-model-name', default='sentence-transformers/bert-base-nli-mean-tokens',
                    help='name of the sentence transformer model')
parser.add_argument('--batch-size', default=64,
                    help='size of mini batch used during training')
parser.add_argument('--epochs', default=20,
                    help='number of epochs')
parser.add_argument('--learning-rate', default=3e-4,
                    help='learning rate used during training')

args = parser.parse_args()

train_paths = args.train_paths.split(',')
test_path = args.test_path
model = args.model
classifier_model_name = args.classifier_model_name
batch_size = args.batch_size
epochs = int(args.epochs)
learning_rate = args.learning_rate


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    if model == 'separate':
        y_pred_1 = evaluate_separate_model(
            device, classifier_model_name, train_paths, test_path, 1, epochs, batch_size, learning_rate)
        y_pred_2 = evaluate_separate_model(
            device, classifier_model_name, train_paths, test_path, 2, epochs, batch_size, learning_rate)
    elif model == 'connected':
        y_pred_1, y_pred_2 = evaluate_connected_model(
            device, classifier_model_name, train_paths, test_path, epochs, batch_size, learning_rate)
    else:
        raise Exception("Model can be only 'separate' or 'connected'!")

    prepare_test_wa_file(test_path=test_path, y_pred_1=y_pred_1, y_pred_2=y_pred_2, filename=f'{model}_results.xml')
