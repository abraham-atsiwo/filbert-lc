from torchtext.vocab import GloVe

from lstm_nsp.lstm_nsp_dataloader import LSTMNSPDataLoader
from pipeline.load_nsp_data import create_nsp_labels

# Load GloVe embeddings
glove = GloVe(name="6B", dim=300)

# Create an embedding layer
embedding_matrix = glove.vectors
# Create a vocabulary dictionary
vocab = glove.stoi

df = create_nsp_labels("data/nsp/bloombery_nsp.parquet", size=5)

loader = LSTMNSPDataLoader(data=df, vocab=vocab, embeddings=embedding_matrix)
print(loader[1])
