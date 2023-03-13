
import time
import argparse
import torch
import pytorch_lightning as pl
import wandb

from stgg.src.util import compute_sequence_accuracy, compute_sequence_cross_entropy
from GDSS.utils.plot import plot_graphs_list

from graph_gen.models.train_evaluate.generator import BaseGenerator
from graph_gen.data import DATASETS
from graph_gen.data.data_utils import train_val_test_split
from graph_gen.data.orderings import order_graphs, ORDER_FUNCS
from graph_gen.data.nx_dataset import GraphRNNDataset, LSTMDataset, GraphDiffusionDataset, GraphAEDataset, LSTMNoTokenDataset, TransNoTokenDataset
from graph_gen.models.model_utils import save_graph_list
from graph_gen.analysis.mmd import evaluate_sampled_graphs


class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        self.ts = time.strftime('%b%d-%H:%M:%S', time.gmtime())
        wandb.config['ts'] = self.ts
        
    def setup_datasets(self, hparams):
        data_name = hparams.dataset_name
        order = hparams.order
        
        # map dataset and split train / test / validation
        graph_getter, num_rep = DATASETS[data_name]
        if data_name == 'zinc250k':
            graphs = graph_getter(zinc_path='resource/zinc.csv')
        elif data_name == 'peptides':
            graphs = graph_getter(peptide_path='')
        else:
            graphs = graph_getter()

        order_func = ORDER_FUNCS[order]
        total_graphs = graphs
        total_ordered_graphs = order_graphs(total_graphs, num_repetitions=num_rep, order_func=order_func, seed=hparams.replicate)
        bw = max([graph.bw for graph in total_ordered_graphs])
        
        self.max_len = max(map(len, graphs)) - 1
        self.train_graphs, self.val_graphs, self.test_graphs = train_val_test_split(graphs)
        
        # map order graphs
        
        ordered_graphs = []
        for graphs in [self.train_graphs, self.val_graphs, self.test_graphs]:
            ordered_graph = order_graphs(graphs, num_repetitions=num_rep, order_func=order_func, seed=hparams.replicate)
            ordered_graphs.append(ordered_graph)
        train_graphs_ord, val_graphs_ord, test_graphs_ord = ordered_graphs
        # set bw
        dataset_cls = {
            "lstm": LSTMDataset,
            "transformer": LSTMDataset,
            "graphrnn": GraphRNNDataset,
            "edpgnn": GraphDiffusionDataset,
            "lstm_notoken": LSTMNoTokenDataset,
            "transformer_notoken": TransNoTokenDataset
        }.get(hparams.model)
        self.bw = bw

        self.total_dataset = dataset_cls([*train_graphs_ord, *test_graphs_ord, *val_graphs_ord], bw, data_name)
        self.train_dataset, self.val_dataset, self.test_dataset = [dataset_cls(graphs, bw, data_name, self.total_dataset, hparams.nhead) for graphs in ordered_graphs]

        
    def setup_model(self, hparams):
        self.model = BaseGenerator(
            num_layers=hparams.num_layers,
            emb_size=hparams.emb_size,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            input_dropout=hparams.input_dropout,
            dropout=hparams.dropout,
        )

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.hparams.lr, 
            )
        
        return [optimizer]

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()

        # decoding
        logits = self.model(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data[0], ignore_index=0)
        statistics["loss/total"] = loss
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data[0], ignore_index=0)[0]

        return loss, statistics

    def training_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
            for key, val in statistics.items():
                self.log(f"train/{key}", val, on_step=False, on_epoch=True, logger=True)
                # wandb.log({f"train/{key}": val})

        return loss

    def validation_step(self, batched_data, batch_idx):
        loss, statistics = self.shared_step(batched_data)
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
            for key, val in statistics.items():
                self.log(f"val/{key}", val, on_step=False, on_epoch=True, logger=True)
            # wandb.log({f"val/{key}": val})
        # pass

    def validation_epoch_end(self, output):
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
            self.check_samples()

    def check_samples(self):
        num_samples = self.hparams.num_samples if not self.trainer.sanity_checking else 2
        sampled_graphs = self.sample(num_samples)
        save_graph_list(self.hparams.dataset_name, self.ts, sampled_graphs)
        plot_dir = f'{self.hparams.dataset_name}/{self.ts}'
        plot_graphs_list(sampled_graphs, save_dir=plot_dir)
        wandb.log({"samples": wandb.Image(f'samples/fig/{plot_dir}/title.png')})
        mmd_results = evaluate_sampled_graphs(sampled_graphs, real_graphs=self.test_graphs[:num_samples])
        wandb.log(mmd_results)
        # wandb.log({"num samples": len(sampled_graphs)})

        
    @staticmethod
    def add_args(parser):

        return parser


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    BaseGeneratorLightningModule.add_args(parser)

    hparams = parser.parse_args()
    
    wandb.init(name=f'{hparams.dataset_name}', project='bwr', group=f'{hparams.group}',
               mode='disabled')
    wandb.config.update(hparams)
    # logger = WandbLogger()

    model = BaseGeneratorLightningModule(hparams)

    wandb.watch(model)
    
    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        gradient_clip_val=hparams.gradient_clip_val
    )
    trainer.fit(model)