from sentence_transformers import SentenceTransformer
from torch import nn


class ClassifierSeparate(nn.Module):
    def __init__(self, labels, model_name='all-mpnet-base-v2'):
        super().__init__()
        if labels == 1:
            self.classes = ['EQUI', 'NOALI', 'OPPO', 'REL', 'SIMI', 'SPE1', 'SPE2']
        else:
            self.classes = ['0', '1', '2', '3', '4', '5', 'NIL']
        self.sbert = SentenceTransformer(model_name)
        self.train_sbert(False)
        self.main = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=768, out_features=500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=300),
            nn.BatchNorm1d(300),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=len(self.classes)),
            nn.Softmax()
        )

    def train_sbert(self, train=True):
        for param in self.sbert.parameters():
            param.requires_grad = train

    def forward(self, ids, mask):
        x = {
            'input_ids': ids,
            'attention_mask': mask
        }
        x = self.sbert(x)['sentence_embedding']
        x = self.main(x)
        return x


class ClassifierConnected(nn.Module):
    def __init__(self, model_name='all-mpnet-base-v2'):
        super().__init__()
        self.classes_1 = ['EQUI', 'NOALI', 'OPPO', 'REL', 'SIMI', 'SPE1', 'SPE2']
        self.classes_2 = ['0', '1', '2', '3', '4', '5', 'NIL']
        self.sbert = SentenceTransformer(model_name)
        self.train_sbert(False)
        self.class_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=768, out_features=500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=300),
            nn.BatchNorm1d(300),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=len(self.classes_1)),
            nn.Softmax()
        )

        self.class_2 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=768, out_features=500),
            nn.BatchNorm1d(500),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=500, out_features=300),
            nn.BatchNorm1d(300),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=300, out_features=100),
            nn.BatchNorm1d(100),
            nn.Dropout(0.2),
            nn.ReLU(),
            nn.Linear(in_features=100, out_features=len(self.classes_2)),
            nn.Softmax()
        )

    def train_sbert(self, train=True):
        for param in self.sbert.parameters():
            param.requires_grad = train

    def forward(self, ids, mask):
        x = {
            'input_ids': ids,
            'attention_mask': mask
        }
        x = self.sbert(x)['sentence_embedding']
        return {
            'class_1': self.class_1(x),
            'class_2': self.class_2(x)
        }
