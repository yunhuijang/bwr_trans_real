import torch
import torch.nn as nn
from torch.distributions import Categorical


class CharRNNGenerator(nn.Module):
    def __init__(
        self, emb_size, dropout, dataset, vocab_size
    ):
        super(CharRNNGenerator, self).__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.dataset = dataset
        self.vocab_size = self.input_size = self.output_size = vocab_size

        self.embedding_layer = nn.Embedding(self.vocab_size, self.emb_size)
        self.lstm_layer = nn.LSTM(self.emb_size, self.emb_size,  
                                dropout=self.dropout, batch_first=True)
        self.linear_layer = nn.Linear(self.emb_size, self.output_size)


    def forward(self, sequences):
        input_pad_seq = sequences[0]
        input_pad_seq = input_pad_seq.squeeze()
        out = self.embedding_layer(input_pad_seq.long())
        out, _ = self.lstm_layer(out)
        out = self.linear_layer(out)
        
        return out    

    def decode(self, num_samples, max_len, device):
        end_token_id = self.vocab_size - 1
        # end_token_id = 1
        padding_token_id = 0
        # 1: start token id
        sequences = torch.LongTensor([[1] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for _ in range(max_len):
            if ended.all():
                break
            logits = self(tuple((sequences, sequences)))
            if _ == 0:
                preds = Categorical(logits=logits).sample()
            else:
                preds = Categorical(logits=logits[:,-1]).sample()
            preds[ended] = padding_token_id
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)
            # TODO: padding token을 예측하면 거기서 종료해야하는지? -> 종료
            ended = torch.logical_or(ended, (preds == end_token_id))
        # add end row (1,0,...,0)
        # sequences = torch.cat([sequences, torch.ones((num_samples,1)).to(device)], dim=1)
        
        # sequences: start token + sequences + end token
        return sequences
