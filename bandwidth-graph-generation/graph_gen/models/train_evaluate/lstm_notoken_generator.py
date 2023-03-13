import torch
import torch.nn as nn
import numpy as np

class LSTMNoTokenGenerator(nn.Module):
    def __init__(
        self, emb_size, dropout, dataset, vocab_size
    ):
        super(LSTMNoTokenGenerator, self).__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.dataset = dataset
        self.vocab_size = self.input_size = self.output_size = vocab_size + 1

        self.lstm_layer = nn.LSTM(self.vocab_size, self.emb_size,  
                                dropout=self.dropout, batch_first=True)
        self.linear_layer = nn.Linear(self.emb_size, self.output_size)


    def forward(self, sequences):
        input_pad_seq = sequences[0]
        input_pad_seq = input_pad_seq.squeeze()
        out, _ = self.lstm_layer(input_pad_seq)
        out = self.linear_layer(out)
        
        return out    

    def decode(self, num_samples, max_len, device):
        padding_row = torch.LongTensor(np.array(np.zeros(self.vocab_size))).to(device)
        start_row = np.array(np.zeros(self.vocab_size))
        start_row[0] = 1
        start_row = start_row.astype(int)
        end_row = start_row
        sequences = torch.FloatTensor([start_row for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for _ in range(max_len):
            if ended.all():
                break
            logits = self(tuple((sequences, sequences)))
            if _ == 0:
                preds = torch.where(logits > 0.5, 1, 0)
                sequences = sequences.unsqueeze(1)
            else:
                preds = torch.where(logits[:, -1] > 0.5, 1, 0)
            preds[ended] = padding_row
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)
            ended = torch.logical_or(ended, torch.Tensor((preds.tolist() == end_row).all(axis=1)).to(device))
        return sequences
