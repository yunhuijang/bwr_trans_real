
import os
# os.chdir('home/yhjang/graph_vocab')

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from moses.moses.utils import disable_rdkit_log, enable_rdkit_log

from model.generator import BaseGenerator
from data.dataset import ZincDataset, MosesDataset, SimpleMosesDataset, QM9Dataset
from data.smiles_dataset import untokenize
from data.target_data import Data
from util import compute_sequence_accuracy, compute_sequence_cross_entropy, canonicalize
from fcd_torch import FCD as FCDMetric
from sklearn.metrics.pairwise import pairwise_kernels
from moses.moses.metrics.metrics import get_all_metrics
import numpy as np
import wandb
import selfies as sf
from GDSS.utils.mol_utils import smiles_to_mols, mols_to_nx, correct_mol
from GDSS.evaluation.mmd import compute_nspdk_mmd
from joblib import Parallel, delayed
from pytorch_lightning.loggers import WandbLogger


class BaseGeneratorLightningModule(pl.LightningModule):
    def __init__(self, hparams):
        super(BaseGeneratorLightningModule, self).__init__()
        hparams = argparse.Namespace(**hparams) if isinstance(hparams, dict) else hparams
        self.save_hyperparameters(hparams)
        self.setup_datasets(hparams)
        self.setup_model(hparams)
        
    def setup_datasets(self, hparams):
        dataset_cls = {
            "zinc": ZincDataset,
            "moses": MosesDataset,
            "simplemoses": SimpleMosesDataset,
            "qm9": QM9Dataset,
        }.get(hparams.dataset_name)
        self.train_dataset = dataset_cls("train_val")
        self.val_dataset = dataset_cls("valid")
        self.test_dataset = dataset_cls("test")
        self.train_smiles_set = set(self.train_dataset.smiles_list)

    def setup_model(self, hparams):
        self.model = BaseGenerator(
            num_layers=hparams.num_layers,
            emb_size=hparams.emb_size,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            input_dropout=hparams.input_dropout,
            dropout=hparams.dropout,
            disable_treeloc=hparams.disable_treeloc,
            disable_graphmask=hparams.disable_graphmask, 
            disable_valencemask=hparams.disable_valencemask,
            enable_absloc=hparams.enable_absloc,
        )

    ### Dataloaders and optimizers
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=Data.collate,
            num_workers=self.hparams.num_workers,
            drop_last=True, 
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=Data.collate,
            num_workers=self.hparams.num_workers,
            drop_last=True,
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
        for key, val in statistics.items():
            self.log(f"train/{key}", val, on_step=True, logger=True)

        return loss

    def validation_step(self, batched_data, batch_idx):
        # loss, statistics = self.shared_step(batched_data)
        # for key, val in statistics.items():
        #     self.log(f"validation/{key}", val, on_step=False, on_epoch=True, logger=True)
        pass

    def validation_epoch_end(self, outputs):
        if (self.current_epoch + 1) % self.hparams.check_sample_every_n_epoch == 0:
            self.check_samples()

    def check_samples(self):
        num_samples = self.hparams.num_samples if not self.trainer.sanity_checking else 2
        smiles_list, results = self.sample(num_samples)

        if self.hparams.string_type == 'selfies':
            train_data = Parallel(n_jobs=8)(delayed(sf.decoder)(selfies) for selfies in self.train_dataset.smiles_list)
            test_data = Parallel(n_jobs=8)(delayed(sf.decoder)(selfies) for selfies in self.test_dataset.smiles_list)
        elif self.hparams.string_type == 'deep_smiles':
            converter = Converter(rings=True, branches=True)
            train_data = Parallel(n_jobs=8)(delayed(converter.decode)(selfies) for selfies in self.train_dataset.smiles_list)
            test_data = Parallel(n_jobs=8)(delayed(converter.decode)(selfies) for selfies in self.test_dataset.smiles_list)
        else:
            train_data = self.train_dataset.smiles_list
            test_data = self.test_dataset.smiles_list
            
        OUTPUT_DIR = "../output"
        valid_smiles_list = [smiles for smiles in smiles_list if smiles is not None]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in train_data]
        if not self.trainer.sanity_checking:
            wandb.init(name=f'{self.hparams.dataset_name}-{self.hparams.string_type}', project='benckmark', group=f'{self.hparams.group}')
            wandb.config.update(self.hparams)
            # calculate FCD
            if len(valid_smiles_list) > 0:
                moses_statistics = get_all_metrics(
                    smiles_list, 
                    n_jobs=self.hparams.num_workers, 
                    device=str(self.device), 
                    train=train_data, 
                    test=test_data,
                    batch_size=256
                )

            # calculate NSPDK
            test_graphs = mols_to_nx(smiles_to_mols(test_data))
            gen_graphs = mols_to_nx(smiles_to_mols([smiles for smiles in smiles_list if smiles is not None]))
            nspdk_score = compute_nspdk_mmd(test_graphs, gen_graphs,
                                            metric='nspdk', is_hist=False, n_jobs=20)
            metrics_dict = moses_statistics
            metrics_dict['NSPDK'] = nspdk_score
            wandb.log(metrics_dict)

            # write generated sample file
            if self.on_epoch_end:
                with open(f'{OUTPUT_DIR}/{self.hparams.dataset_name}_{num_samples}_{self.hparams.string_type}_{self.hparams.is_pretokenized}_{self.hparams.is_character}_list.txt', 'w') as f :
                    for smiles in smiles_list:
                        if smiles is not None:
                            f.write("%s\n" %smiles)


    def sample(self, num_samples):
        offset = 0
        results = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            self.model.eval()
            with torch.no_grad():
                data_list = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)

            results.extend((data.to_smiles(), "".join(data.tokens), data.error) for data in data_list)
        
        disable_rdkit_log()
        smiles_list = [canonicalize(elem[0]) for elem in results]
        enable_rdkit_log()

        return smiles_list, results
        
    @staticmethod
    def add_args(parser):
        #
        parser.add_argument("--dataset_name", type=str, default="zinc")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=6)

        #
        parser.add_argument("--num_layers", type=int, default=6)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--input_dropout", type=float, default=0.0)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--logit_hidden_dim", type=int, default=256)
        
        #
        parser.add_argument("--disable_treeloc", action="store_true")
        parser.add_argument("--disable_graphmask", action="store_true")
        parser.add_argument("--disable_valencemask", action="store_true")
        parser.add_argument("--enable_absloc", action="store_true")
        
        #
        parser.add_argument("--lr", type=float, default=2e-4)

        #
        parser.add_argument("--max_len", type=int, default=250)
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=1)
        parser.add_argument("--num_samples", type=int, default=10000)
        parser.add_argument("--sample_batch_size", type=int, default=1000)
        parser.add_argument("--eval_moses", action="store_true")

        parser.add_argument("--string_type", type=str, default='smiles')
        parser.add_argument("--is_character", type=str, default='token')
        parser.add_argument("--is_pretokenized", type=str, default='wopre')
        

        return parser


if __name__ == "__main__":
    
    # wandb.init(name='QM9-STGG')
    parser = argparse.ArgumentParser()
    BaseGeneratorLightningModule.add_args(parser)

    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--gradient_clip_val", type=float, default=1.0)
    parser.add_argument("--load_checkpoint_path", type=str, default="")
    parser.add_argument("--resume_from_checkpoint_path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--group", type=str, default='char_rnn')

    hparams = parser.parse_args()
    # wandb.config.update(hparams)

    model = BaseGeneratorLightningModule(hparams)
    # wandb.watch(model)

    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        gradient_clip_val=hparams.gradient_clip_val,
        resume_from_checkpoint=hparams.resume_from_checkpoint_path
    )
    trainer.fit(model)
    # wandb.finish()