from sklearn.metrics import classification_report
from torch import nn, optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from src.models.classifiers import ClassifierSeparate, ClassifierConnected
from src.models.datasets import MyDataset, MyDatasetConnected
from src.operations.training import train_model, train_model_connected, get_accuracy, get_accuracy_connected
from src.processing.data import get_df, collate_fn


def load_dfs(paths, labels):
    df_connected, y_connected = pd.DataFrame(), pd.Series()

    for path in paths:
        df, y = get_df(path, labels)
        df_connected = df_connected.append(df).reset_index(drop=True)
        y_connected = y_connected.append(y).reset_index(drop=True)
        print(path)
        print(df_connected.shape)

    return df_connected, y_connected


def evaluate_separate_model(device, classifier_model_name, train_paths, test_path, labels, epochs, batch_size, learning_rate):
    df_train, y_train = load_dfs(train_paths, labels=labels)
    df_test, y_test = get_df(test_path, labels=labels)

    train_dataset = MyDataset(df_train['text_source'].values, df_train['text_translation'].values, y_train.values, labels)
    test_dataset = MyDataset(df_test['text_source'].values, df_test['text_translation'].values, y_test.values, labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model_separate = ClassifierSeparate(model_name=classifier_model_name, labels=labels)
    model_separate = model_separate.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_separate = optim.Adam(model_separate.parameters(), lr=learning_rate)

    train_model(
        model=model_separate,
        optimizer=optimizer_separate,
        epochs=epochs,
        train_dataloader=train_dataloader,
        criterion=criterion,
        device=device,
        plot_filename=f'separate_losses_{labels}',
    )

    acc, y_true, y_pred = get_accuracy(model_separate, test_dataloader, device)
    print(classification_report(y_true, y_pred, target_names=list(np.unique(y_test))))
    return y_pred


def evaluate_connected_model(device, classifier_model_name, train_paths, test_path, epochs, batch_size, learning_rate):
    df_train, y_train_1 = load_dfs(train_paths, labels=1)
    df_test, y_test_1 = get_df(test_path, labels=1)

    _, y_train_2 = load_dfs(train_paths, labels=2)
    _, y_test_2 = get_df(test_path, labels=2)

    train_dataset = MyDatasetConnected(df_train['text_source'].values, df_train['text_translation'].values,
                                       y_train_1.values, y_train_2.values)
    test_dataset = MyDatasetConnected(df_test['text_source'].values, df_test['text_translation'].values,
                                      y_test_1.values, y_test_2.values)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True,
                                  collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model_connected = ClassifierConnected(model_name=classifier_model_name)
    model_connected.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_connected = optim.Adam(model_connected.parameters(), lr=learning_rate)

    train_model_connected(model_connected, optimizer_connected, epochs, train_dataloader, criterion, device)
    acc_1, acc_2, y_true_1, y_true_2, y_pred_1, y_pred_2 = get_accuracy_connected(model_connected, test_dataloader,
                                                                                  device)

    return y_pred_1, y_pred_2