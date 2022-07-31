import re

import pandas as pd


def prepare_data(train_dir, test_dir, datasets):
    for i in ['train', 'test']:
        for dataset in datasets:
            if i == 'test':
                path = f'{test_dir}/STSint.testinput.{dataset}.wa'
                path_processed = f'{test_dir}/processed_{dataset}.xml'
            else:
                path = f'{train_dir}/STSint.input.{dataset}.wa'
                path_processed = f'{train_dir}/processed_{dataset}.xml'

            process_dataset(path, path_processed)


def process_dataset(path, path_processed):
    with open(path, 'rb') as reader:
        with open(path_processed, 'wb') as writer:
            text = reader.read()
            text = text.replace(b'<==>', b'---')
            text = text.replace(b'&', b'und')
            writer.write(text)

    start_root = '<root>'
    end_root = '</root>'

    with open(path_processed, "a+") as f:
        f.write(end_root)

    with open(path_processed, "r+") as f:
        content = f.read()
        f.seek(0)
        f.write(start_root + '\n' + content)


# Generator dla klasy 1
def gen_1(preds_1):
    for i in preds_1:
        yield i

# Generator dla klasy 2
def gen_2(preds_2):
    for i in preds_2:
        yield i


def to_correct_class(idx, class_type):
    type_1 = ['EQUI', 'NOALI', 'OPPO', 'REL', 'SIMI', 'SPE1', 'SPE2']
    type_2 = ['0', '1', '2', '3', '4', '5', 'NIL']
    if class_type == 1:
        return type_1[idx]
    else:
        return type_2[idx]


def modify_alignment(text, g_1, g_2):
    chunks = text.split('\n')
    new_chunks = []
    for chunk in chunks:
        # Np. do NIL, 0, 1, ..., 5
        chunk = re.sub(r'\s\/\/\s(\bNIL\b)*[0-6]*\s\/\/\s', f' // {to_correct_class(next(g_2), 2)} // ', chunk)
        # Np. do EQUI, SPE1, SPE2, ...
        chunk = re.sub(r'\s\/\/\s[A-Z]+[1-2]*\s\/\/\s', f' // {to_correct_class(next(g_1), 1)} // ', chunk)
        # Jak NIL to musi być NOALI
        chunk = re.sub(r'\s\/\/\s(.*?)\s\/\/\s(\bNIL\b)\s\/\/\s', f' // NOALI // 0 // ', chunk)
        # A Jak NOALI to musi być 0
        chunk = re.sub(r'\s\/\/\s(\bNOALI\b)\s\/\/\s(.*?)\s\/\/\s', f' // NOALI // 0 // ', chunk)
        new_chunks.append(chunk)
    text = '\n'.join(new_chunks)

    return text


def get_rows_with_nan(df):
    rows_with_nan = []
    for index, row in df.iterrows():
        is_nan_series = row.isnull()
        if is_nan_series.any():
            rows_with_nan.append(index)
    return rows_with_nan


def prepare_test_wa_file(test_path, y_pred_1, y_pred_2, filename):
    g_1 = gen_1(y_pred_1)
    g_2 = gen_2(y_pred_2)

    df = pd.read_xml(test_path)
    df.drop('status', axis=1, inplace=True)
    df.drop('id', axis=1, inplace=True)
    rows_with_nan = get_rows_with_nan(df)
    rows_with_nan = [x+1 for x in rows_with_nan]
    df.dropna(axis=0, how='any', inplace=True)

    df['alignment'] = df['alignment'].apply(modify_alignment, args=(g_1, g_2))
    df.to_xml(filename, index=False, row_name='Sentence', root_name='root', xml_declaration=False, pretty_print=True)
    text = ''
    tags = ['root', 'sentence', 'source', 'translation', 'alignment']
    with open(filename, 'r') as file:
        text = file.read()
        text = text.replace('<root>', '')
        text = text.replace('</root>', '')
        text = text.replace('<sentence>', '')
        text = text.replace('</sentence>', '')
        text = text.replace('<Sentence>', '<sentence>')
        text = text.replace('</Sentence>', '</sentence>')
        text = text.replace('---', '<==>')
        text = text.replace('und', '&')
        for tag in tags:
            text = text.replace(f'<{tag}>', f'\n<{tag}>\n')
            text = text.replace(f'</{tag}>', f'\n</{tag}>\n')
        text = text.strip()
        text = "\n".join([ll.rstrip() for ll in text.splitlines() if ll.strip()])
    with open(filename, 'w') as file:
        file.write(text)

    with open(filename, 'r') as file:
        lines = file.readlines()
        index = 1
        new_lines = []
        for line in lines:
            if '<sentence>' in line:
                if index in rows_with_nan:
                    index += 1
                line = line.replace('<sentence>', f'<sentence id=\"{index}\" status=\"\">')
                index += 1
            if '</sentence>' in line:
                line = line.replace('</sentence>', '</sentence>\n\n')
            new_lines.append(line)
    with open(filename, 'w') as file:
        file.writelines(new_lines)
