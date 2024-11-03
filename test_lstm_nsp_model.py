from lstm_nsp.lstm_nsp_model import LSTMNSPModel
from lstm_nsp.lstm_nsp_dataloader import LSTMNSPDataLoader
from torch.utils.data import DataLoader
import torch.nn as nn 
import torch

from pipeline.load_nsp_data import create_nsp_labels
from torchtext.vocab import GloVe


def collate_fn(batch):
    pad_index = 0
    batch_x1, batch_x2, batch_y = zip(*batch)
    max_length_x1 = max(len(x) for x in batch_x1)
    max_length_x2 = max(len(x) for x in batch_x2)
    
    padded_x1 = [list(x) + [pad_index] * (max_length_x1 - len(x)) for x in batch_x1]
    padded_x2 = [list(x) + [pad_index] * (max_length_x2 - len(x)) for x in batch_x2]
    
    padded_x1 = torch.tensor(padded_x1, dtype=torch.long)
    padded_x2 = torch.tensor(padded_x2, dtype=torch.long)
    batch_y = torch.tensor(batch_y, dtype=torch.float)
    
    return padded_x1, padded_x2, batch_y

glove = GloVe(name="6B", dim=300)
# Create an embedding layer
embedding_matrix = glove.vectors
# Create a vocabulary dictionary
vocab = glove.stoi
df = create_nsp_labels("data/nsp/bloombery_nsp.parquet", size=50)

loader = LSTMNSPDataLoader(data=df, vocab=vocab, embeddings=embedding_matrix)

model = LSTMNSPModel(
    embedding_type="fasttext", hidden_dim=128, output_dim=2, glove_dim=100, freeze=False
)


dataloader = DataLoader(loader, batch_size=8, shuffle=True, collate_fn=collate_fn)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


for data in dataloader:
    print(data)