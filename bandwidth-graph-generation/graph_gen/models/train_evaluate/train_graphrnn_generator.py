import argparse

import torch
from torch.utils.data import DataLoader
import wandb
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger


from train_generator import BaseGeneratorLightningModule
from graph_gen.data.nx_dataset import GraphRNNDataset
from graph_gen.models.graph_rnn import GraphRNNSimple
from graph_gen.data.data_utils import adj_to_graph, bw_matrix_to_adj


class GraphRNNGeneratorLightningModule(BaseGeneratorLightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
        self.max_len = self.train_dataset.band_flat_adjs[0][0].shape[0] - 1
        self.temperature = hparams.temperature
    
    def setup_model(self, hparams):
        self.model = GraphRNNSimple(
            hidden_dim=hparams.emb_size,
            epochs=hparams.max_epochs,
            bw=self.bw,
            lr=hparams.lr,
            wd=hparams.wd,
        )
        self.model.to('cuda:0')
        
    
    ### Main steps
    def shared_step(self, batched_data):
        
        loss, statistics = 0.0, dict()
        packed_input, packed_output = batched_data
        pred = self.model(packed_input)
        loss = self.model.loss(pred.data, packed_output.data)
        statistics["loss/total"] = loss
        
        return loss, statistics

    def generate_samples(self, num_samples):
        x = torch.zeros((num_samples, 1, self.bw + 1), device=self.device)
        x[:, 0, 0] = 1  # start token
        for _ in range(self.max_len):
            next_x = self.model.graph_rnn.unpacked_forward(x)
            next_p = torch.sigmoid(next_x[:, -1] * 1 / self.temperature)
            sampled = torch.bernoulli(next_p).unsqueeze(1)
            x = torch.cat([x, sampled], dim=1)
            
        return x
    
    def sample(self, num_samples):
        # generate samples
        offset = 0
        graphs = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples
            with torch.no_grad():
                x = self.generate_samples(cur_num_samples)
                for sample in x:
                    sampled_no_start_token = sample[1:]
                    adj = bw_matrix_to_adj(sampled_no_start_token)
                    graph = adj_to_graph(adj)
                    graphs.append(graph)
        return graphs

        ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=GraphRNNDataset.collate_fn,
            num_workers=self.hparams.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=GraphRNNDataset.collate_fn,
            num_workers=self.hparams.num_workers
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=GraphRNNDataset.collate_fn,
            num_workers=self.hparams.num_workers
        )
    
    
    @staticmethod
    def add_args(parser):
        
        parser.add_argument("--dataset_name", type=str, default="planar")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=6)
        parser.add_argument("--order", type=str, default='C-M')
        parser.add_argument("--replicate", type=int, default=0)

        #
        parser.add_argument("--model", type=str, default='graphrnn')
        parser.add_argument("--emb_size", type=int, default=128)

        #
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--gradient_clip_val", type=int, default=1)
        
        #
        parser.add_argument("--num_samples", type=int, default=100)
        parser.add_argument("--sample_batch_size", type=int, default=1000)
        
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=1)
        parser.add_argument("--max_epochs", type=int, default=10)
        parser.add_argument("--group", type=str, default='graphrnn')
        
        # GraphRNN
        parser.add_argument("--wd", type=float, default=0)
        parser.add_argument("--temperature", type=float, default=0.4)
        
        parser.add_argument("--wandb_on", type=str, default='disabled')

        return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    GraphRNNGeneratorLightningModule.add_args(parser)

    hparams = parser.parse_args()
    
    wandb_logger = WandbLogger(name=f'{hparams.dataset_name}-{hparams.model}', 
                               project='bwr', group=f'{hparams.group}', mode=f'{hparams.wandb_on}')
 
    wandb.config.update(hparams)
    
    model = GraphRNNGeneratorLightningModule(hparams)
    
    wandb.watch(model)
    
    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        logger=wandb_logger
    )
    trainer.fit(model)