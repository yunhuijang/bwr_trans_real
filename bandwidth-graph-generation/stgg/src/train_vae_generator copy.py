import os
import argparse
from moses.moses.vae.model import VAE

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from neptune.new.integrations.pytorch_lightning import NeptuneLogger

from moses.utils import disable_rdkit_log, enable_rdkit_log
from moses.metrics.metrics import get_all_metrics
from moses.aae.model import AAE
from moses.vae.model import VAE
from moses.organ.model import ORGAN
# from moses.latentgan.model import LatentGAN

from moses.moses.aae.trainer import AAETrainer
from moses.vae.config import get_parser as VAEParser
from moses.organ.config import get_config as ORGANConfig
# from moses.latentgan.model import get_parser as LatentGANParser
from moses.aae.config import get_parser as AAEParser
from moses.script_utils import add_train_args
from moses.models_storage import ModelsStorage

from train_generator import BaseGeneratorLightningModule
from data.smiles_dataset import ZincDataset, MosesDataset, SimpleMosesDataset, QM9Dataset, untokenize
from util import compute_sequence_accuracy, compute_sequence_cross_entropy, canonicalize
from fcd_torch import FCD as FCDMetric
from eden.graph import vectorize
from sklearn.metrics.pairwise import pairwise_kernels
import numpy as np
import wandb
import pickle
import sys
from GDSS.utils.mol_utils import smiles_to_mols, mols_to_nx
from GDSS.evaluation.mmd import compute_nspdk_mmd

class MosesGeneratorLightningModule(BaseGeneratorLightningModule):
    def setup_datasets(self, hparams):
        dataset_cls = {
            "zinc": ZincDataset,
            "moses": MosesDataset,
            "simplemoses": SimpleMosesDataset,
            "qm9": QM9Dataset,
        }.get(hparams.dataset_name)
        self.train_dataset = dataset_cls("train")
        self.val_dataset = dataset_cls("valid")
        self.test_dataset = dataset_cls("test")
        self.train_smiles_set = set(self.train_dataset.smiles_list)


    def setup_model(self, hparams):
        model_cls = {
            "aae": AAE,
            "vae": VAE,
            "organ": ORGAN,
            # "latentgan": LatentGAN
        }.get(hparams.model)
        MODELS = ModelsStorage()
        trainer = MODELS.get_model_trainer(hparams.model)(hparams)
        self.model = model_cls(
            vocabulary=trainer.get_vocabulary(data=self.train_dataset),
            config=hparams
            )

    ### 
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=True,
            collate_fn=lambda sequences: pad_sequence(sequences, batch_first=True, padding_value=0),
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            shuffle=False,
            collate_fn=lambda sequences: pad_sequence(sequences, batch_first=True, padding_value=0),
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )

    ### Main steps
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        logits = self.model(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data, ignore_index=0)
        statistics["loss/total"] = loss
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data, ignore_index=0)[0]

        return loss, statistics

    
    # 
    def check_samples(self):
        num_samples = self.hparams.num_samples if not self.trainer.sanity_checking else 2
        smiles_list, results = self.sample(num_samples)
        OUTPUT_DIR = "../output"
        valid_smiles_list = [smiles for smiles in smiles_list if smiles is not None]
        unique_smiles_set = set(valid_smiles_list)
        novel_smiles_list = [smiles for smiles in valid_smiles_list if smiles not in self.train_smiles_set]

        if not self.trainer.sanity_checking:
            # calculate FCD
            if len(valid_smiles_list) > 0:
                moses_statistics = get_all_metrics(
                    smiles_list, 
                    n_jobs=self.hparams.num_workers, 
                    device=str(self.device), 
                    train=self.train_dataset.smiles_list, 
                    test=self.test_dataset.smiles_list,
                )
            # calculate NSPDK
            test_graphs = mols_to_nx(smiles_to_mols(self.test_dataset.smiles_list))
            gen_graphs = mols_to_nx(smiles_to_mols(results))
            nspdk_score = compute_nspdk_mmd(test_graphs, gen_graphs,
                                            metric='nspdk', is_hist=False, n_jobs=20)
            metrics_dict = moses_statistics
            metrics_dict['NSPDK'] = nspdk_score
            wandb.log(metrics_dict)

            # write generated sample file
            if self.on_epoch_end:
                with open(f'{OUTPUT_DIR}/{hparams.dataset_name}_{num_samples}_{hparams.model}_list.txt', 'wb') as f :
                    pickle.dump(results, f)

    def sample(self, num_samples):
        offset = 0
        results = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            self.model.eval()
            with torch.no_grad():
                sequences = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)

            results.extend(untokenize(sequence, string_type=self.hparams.string_type) for sequence in sequences.tolist())

        disable_rdkit_log()
        smiles_list = list(map(canonicalize, results))
        enable_rdkit_log()

        return smiles_list, results

    @staticmethod
    def add_args(parser):

        parser.add_argument("--model", type=str, default='vae')
        #
        parser.add_argument("--dataset_name", type=str, default="qm9")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=6)

        #
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--emb_size", type=int, default=1024)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--input_dropout", type=int, default=0.0)
        parser.add_argument("--dropout", type=int, default=0.1)
        parser.add_argument("--logit_hidden_dim", type=int, default=256)

        #
        # parser.add_argument("--lr", type=float, default=2e-4)
        
        #
        parser.add_argument("--max_len", type=int, default=250)
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=1)
        parser.add_argument("--num_samples", type=int, default=10000)
        parser.add_argument("--sample_batch_size", type=int, default=1000)
        parser.add_argument("--eval_moses", action="store_true")

        #
        parser.add_argument("--string_type", type=str, default='smiles')

        return parser


if __name__ == "__main__":
    wandb.init()

    parser = VAEParser()
    add_train_args(parser)
    MosesGeneratorLightningModule.add_args(parser)
    # parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--tag", type=str, default="default")
    hparams = parser.parse_args()
    wandb.config.update(hparams)

    MODELS = ModelsStorage()
    model = MosesGeneratorLightningModule(hparams)
    wandb.watch(model)
    trainer = MODELS.get_model_trainer(hparams.model)(hparams)

    trainer.fit(model.model, model.train_dataset, model.test_dataset)