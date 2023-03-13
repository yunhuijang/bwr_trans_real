import torch
import torch.nn as nn
from torch.distributions import Categorical
import math


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0)]
        return x

class TransformerGenerator(nn.Module):
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, input_dropout, dropout, vocab_size, max_len, pos_abs=False
    ):
        super(TransformerGenerator, self).__init__()
        self.nhead = nhead
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.pos_abs = pos_abs
        
        #
        self.token_embedding_layer = TokenEmbedding(vocab_size, emb_size)
        
        self.positional_encoding = PositionalEncoding(emb_size, max_len=max_len)
        
        self.input_dropout = nn.Dropout(input_dropout)

        self.distance_embedding_layer = nn.Embedding(max_len + 1, nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(emb_size, nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(emb_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)

        #
        self.generator = nn.Linear(emb_size, vocab_size)

    def forward(self, sequences):
        
        sequences = sequences[0]
        batch_size = sequences.size(0)
        sequence_len = sequences.size(1)
        sequences = sequences.squeeze(2)
        
        out = self.token_embedding_layer(sequences)
        if self.pos_abs:
            out = self.positional_encoding(out)
            mask = torch.zeros(batch_size, sequence_len, sequence_len, self.nhead, device=out.device)
        out = self.input_dropout(out)

        if not self.pos_abs:
            # distance_squares: distance between each pair
            distance_squares = torch.abs(torch.arange(sequence_len).unsqueeze(0) - torch.arange(sequence_len).unsqueeze(1))
            distance_squares[distance_squares > self.max_len] = self.max_len
            distance_squares = distance_squares.unsqueeze(0).repeat(batch_size, 1, 1)
            distance_squares = distance_squares.to(out.device)
            # mask: embedding of distances
            mask = self.distance_embedding_layer(distance_squares)
        mask = mask.permute(0, 3, 1, 2)

        # bool_mask: mask future values
        bool_mask = (torch.triu(torch.ones((sequence_len, sequence_len))) == 1).transpose(0, 1)
        bool_mask = bool_mask.view(1, 1, sequence_len, sequence_len).repeat(batch_size, self.nhead, 1, 1).to(out.device)
        mask = mask.masked_fill(bool_mask == 0, float("-inf"))
        mask = mask.reshape(-1, sequence_len, sequence_len)

        #
        key_padding_mask = sequences == 0

        out = out.transpose(0, 1)
        out = self.transformer(out, mask, key_padding_mask)
        out = out.transpose(0, 1)

        #
        logits = self.generator(out)

        return logits

    def decode(self, num_samples, max_len, device):
        # 1: start token
        sequences = torch.LongTensor([[1] for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for _ in range(max_len):
            if ended.all():
                break
            sequences = sequences.unsqueeze(2)
            logits = self(tuple((sequences, sequences)))
            if _ == 0:
                preds = Categorical(logits=logits).sample()
            else:
                preds = Categorical(logits=logits[:,-1]).sample()
            # 0: pad token
            preds = preds.squeeze()
            preds[ended] = 0
            sequences = sequences.squeeze(2)
            sequences = torch.cat([sequences, preds.unsqueeze(1)], dim=1)
            # 1: end token
            ended = torch.logical_or(ended, preds == 1)
        sequences = torch.cat([sequences, torch.ones((num_samples,1)).to(device)], dim=1)
        
        return sequences
