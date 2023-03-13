import numpy as np
import torch
import torch.nn as nn
import math


# helper Module to convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


class TransformerNoTokenGenerator(nn.Module):
    def __init__(
        self, num_layers, emb_size, nhead, dim_feedforward, input_dropout, dropout, vocab_size, max_len, pad_vocab_size
    ):
        super(TransformerNoTokenGenerator, self).__init__()
        self.nhead = nhead
        self.vocab_size = vocab_size + 1
        self.max_len = max_len
        self.pad_vocab_size = pad_vocab_size
        
        self.input_dropout = nn.Dropout(input_dropout)

        self.distance_embedding_layer = nn.Embedding(self.max_len + 1, self.nhead)

        #
        encoder_layer = nn.TransformerEncoderLayer(self.pad_vocab_size, self.nhead, dim_feedforward, dropout, "gelu")
        encoder_norm = nn.LayerNorm(self.pad_vocab_size)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        

    def forward(self, sequences):
        
        sequences = sequences[0]
        batch_size = sequences.size(0)
        # training
        if len(sequences.shape) == 4:
            sequence_len = sequences.size(1)
            sequences = sequences.squeeze(2)
        # sampling
        else:
            sequence_len = sequences.size(2)
            sequences = sequences.transpose(1,2)
            
        
        # out = self.token_embedding_layer(sequences)
        out = self.input_dropout(sequences)

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
        key_padding_mask = torch.logical_not(sequences.any(dim=2))

        out = out.transpose(0, 1)
        logits = self.transformer(out, mask, key_padding_mask)
        logits = logits.transpose(0, 1)


        return logits

    def decode(self, num_samples, max_len, device):
        padding_row = torch.LongTensor(np.array(np.zeros(self.pad_vocab_size))).to(device)
        start_row = np.array(np.zeros(self.pad_vocab_size))
        start_row[0] = 1
        start_row = start_row.astype(int)
        end_row = start_row
        sequences = torch.FloatTensor([start_row for _ in range(num_samples)]).to(device)
        ended = torch.tensor([False for _ in range(num_samples)], dtype=torch.bool).to(device)
        for _ in range(max_len):
            if ended.all():
                break
            logits = self(tuple((sequences.unsqueeze(2), sequences.unsqueeze(2))))
            if _ == 0:
                preds = torch.where(logits > 0.5, 1, 0)
                preds[ended] = padding_row
                sequences = torch.cat([sequences.unsqueeze(1), preds], dim=1)
            else:
                preds = torch.where(logits[:, -1] > 0.5, 1, 0).unsqueeze(1)
                preds[ended] = padding_row
                sequences = torch.cat([sequences, preds], dim=1)
        
            
            ended = torch.logical_or(ended, torch.Tensor((preds.tolist() == end_row)).all(dim=2).to(device).squeeze())
        return sequences
