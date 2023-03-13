import argparse

import torch
from torch.utils.data import DataLoader
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from train_generator import BaseGeneratorLightningModule
from lstm_generator import CharRNNGenerator
from graph_gen.data.nx_dataset import LSTMDataset
from graph_gen.models.model_utils import compute_sequence_accuracy, compute_sequence_cross_entropy


class LSTMGeneratorLightningModule(BaseGeneratorLightningModule):
    def __init__(self, hparams):
        self.setup_datasets(hparams)
        self.setup_vocab_size()
        super().__init__(hparams)
        self.max_len = self.train_dataset.band_flat_adjs[0][0].shape[0] - 1
    
    def setup_vocab_size(self):
        self.vocab_size = len(self.train_dataset.token_to_id) + 2
    
    def setup_model(self, hparams):
        self.model = CharRNNGenerator(
            emb_size=hparams.emb_size,
            dropout=hparams.dropout,
            dataset=hparams.dataset_name,
            vocab_size=self.vocab_size
        )
        self.model.to('cuda:0')
    
    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        logits = self.model(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data[1], ignore_index=0)
        statistics["loss/total"] = loss
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data[1], ignore_index=0)[0]
        
        return loss, statistics

    def sample(self, num_samples):
        offset = 0
        graphs = []
        token_to_id = self.train_dataset.token_to_id
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            self.model.eval()
            with torch.no_grad():
                sequences = self.model.decode(cur_num_samples, max_len=self.max_len, device=self.device)
            graphs_list = [LSTMDataset.untokenize(sequence, token_to_id, self.vocab_size) for sequence in sequences.tolist()]
            graphs.extend(graphs_list)
            # filter out None
            graphs = [graph for graph in graphs if graph is not None]

        return graphs

        ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=LSTMDataset.collate_fn,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=LSTMDataset.collate_fn,
            num_workers=self.hparams.num_workers
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=LSTMDataset.collate_fn,
            num_workers=self.hparams.num_workers
        )
    
    
    @staticmethod
    def add_args(parser):
        
        parser.add_argument("--dataset_name", type=str, default="ENZYMES")
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--num_workers", type=int, default=6)
        parser.add_argument("--order", type=str, default='C-M')
        parser.add_argument("--replicate", type=int, default=0)

        #
        parser.add_argument("--model", type=str, default='lstm')
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--input_dropout", type=int, default=0.0)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--logit_hidden_dim", type=int, default=256)

        #
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--gradient_clip_val", type=int, default=1)
        
        #
        parser.add_argument("--num_samples", type=int, default=100)
        parser.add_argument("--sample_batch_size", type=int, default=1000)
        
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=1)
        parser.add_argument("--max_epochs", type=int, default=10)
        parser.add_argument("--group", type=str, default='lstm')
        
        # wandb
        
        parser.add_argument("--wandb_on", type=str, default="disabled")

        return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    LSTMGeneratorLightningModule.add_args(parser)

    hparams = parser.parse_args()
    
    wandb_logger = WandbLogger(name=f'{hparams.dataset_name}-{hparams.model}', 
                               project='bwr', group=f'{hparams.group}', mode=f'{hparams.wandb_on}')
    
    wandb.config.update(hparams)
    
    model = LSTMGeneratorLightningModule(hparams)
    
    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        logger=wandb_logger
    )
    trainer.fit(model)