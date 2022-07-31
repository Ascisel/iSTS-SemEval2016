from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, chunk_source, chunk_translation, targets, class_):
        self.chunk_source = chunk_source
        self.chunk_translation = chunk_translation
        self.pairs = [chunk_source, chunk_translation]
        self.targets = targets
        if class_ == 1:
            self.classes = ['EQUI', 'NOALI', 'OPPO', 'REL', 'SIMI', 'SPE1', 'SPE2']
        else:
            self.classes = ['0', '1', '2', '3', '4', '5', 'NIL']

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        chunks = (self.pairs[0][idx], self.pairs[1][idx])
        target = self.classes.index(self.targets[idx])
        return chunks, target


class MyDatasetConnected(Dataset):
    def __init__(self, chunk_source, chunk_translation, targets_1, targets_2):
        self.chunk_source = chunk_source
        self.chunk_translation = chunk_translation
        self.pairs = [chunk_source, chunk_translation]
        self.targets_1 = targets_1
        self.targets_2 = targets_2
        self.classes_1 = ['EQUI', 'NOALI', 'OPPO', 'REL', 'SIMI', 'SPE1', 'SPE2']
        self.classes_2 = ['0', '1', '2', '3', '4', '5', 'NIL']

    def __len__(self):
        return len(self.targets_1)

    def __getitem__(self, idx):
        return (self.pairs[0][idx], self.pairs[1][idx]), \
               (self.classes_1.index(self.targets_1[idx]), self.classes_2.index(self.targets_2[idx]))
