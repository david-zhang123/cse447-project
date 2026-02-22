import torch
import torch.nn as nn
from torch.utils.data import Dataset

class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.chars = sorted(list(set(text)))
        self.char_to_idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.data = [self.char_to_idx[c] for c in text]
        self.seq_len = seq_len
        self.num_samples = len(self.data) - seq_len

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[-1], dtype=torch.long)
        return x, y

class SimpleLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(SimpleLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        # We only care about the last output for next-char prediction
        last_output = output[:, -1, :] 
        logits = self.fc(last_output)
        return logits
