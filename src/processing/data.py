from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pandas as pd
import torch

from src.processing.text import clear_alignment, clear_sentence, source_to_words


def df_from_alignment(series):
    new_df = pd.DataFrame(columns=[
        'class_1', 'class_2', 'chunks_source', 'chunks_translation', 'text_source', 'text_translation'
    ])

    for data in series:
        for element in data:
            sentence = element[4]
            row = {
                'class_1': element[0],
                'class_2': element[1],
                'chunks_source': element[2],
                'chunks_translation': element[3],
                'text_source': sentence[0],
                'text_translation': sentence[1],
            }

            new_df = new_df.append(row, ignore_index=True)

    return new_df


def get_classes_weight(df_labels):
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(df_labels), y=np.array(df_labels))
    class_weights = torch.tensor(class_weights, dtype=torch.float)
    return class_weights


def collate_fn(batch):
    data_list, label_list = [], []
    for _data, _label in batch:
        data_list.append(_data)
        label_list.append(_label)
    return data_list, torch.LongTensor(label_list)


def get_df(path, labels=1):
    df = pd.read_xml(path)
    df.drop('status', axis=1, inplace=True)

    df['source_split'] = df['source'].apply(source_to_words)
    df['translation_split'] = df['translation'].apply(source_to_words)

    split_columns = df['sentence'].str.split(pat='\n', expand=True)
    df['source'] = split_columns[0]
    df['translation'] = split_columns[1]
    df.drop('sentence', axis=1, inplace=True)

    df['source'] = df['source'].apply(clear_sentence)
    df['translation'] = df['translation'].apply(clear_sentence)
    df.dropna(how='any', axis=0, inplace=True)
    df['alignment'] = df['alignment'].apply(clear_alignment)

    new_df = df_from_alignment(df['alignment'])

    def change_classes(text):
        class_1_change = {
            'SPE1_FACT': 'SPE1',
            'SIMI_FACT': 'SIMI',
            'EQUI_POL': 'EQUI',
            'EQUI_FACT': 'EQUI',
            'REL_POL': 'REL',
            'SPE2_FACT': 'SPE2',
            'NOALI_FACT': 'NOALI',
            'SPE2_POL': 'SPE2'
        }
        try:
            new_class_name = class_1_change[text]
            return new_class_name
        except KeyError as e:
            return text

    new_df['class_1'] = new_df['class_1'].apply(change_classes)
    df_labels_1 = new_df.pop('class_1')
    df_labels_2 = new_df.pop('class_2')

    new_df.drop(columns=['chunks_source', 'chunks_translation'], inplace=True)

    if labels == 1:
        df_labels = df_labels_1
    else:
        df_labels = df_labels_2
    return new_df, df_labels
