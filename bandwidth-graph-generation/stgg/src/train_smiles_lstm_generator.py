import argparse

import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import pytorch_lightning as pl

from moses.utils import disable_rdkit_log, enable_rdkit_log
from model.smiles_lstm_generator import CharRNNGenerator
from train_generator import BaseGeneratorLightningModule
from data.smiles_dataset import ZincDataset, MosesDataset, SimpleMosesDataset, QM9Dataset, untokenize
from util import canonicalize_deep_smiles, compute_sequence_accuracy, compute_sequence_cross_entropy, canonicalize, canonicalize_selfies, canonicalize_deep_smiles
import selfies as sf
from deepsmiles import Converter
from joblib import Parallel, delayed


class SmilesCharGeneratorLightningModule(BaseGeneratorLightningModule):
    def setup_datasets(self, hparams):
        dataset_cls = {
            "zinc": ZincDataset,
            "moses": MosesDataset,
            "simplemoses": SimpleMosesDataset,
            "qm9": QM9Dataset,
        }.get(hparams.dataset_name)
        self.train_dataset = dataset_cls("train_val", hparams.string_type)
        self.val_dataset = dataset_cls("valid", hparams.string_type)
        self.test_dataset = dataset_cls("test", hparams.string_type)
        self.train_smiles_set = set(self.train_dataset.smiles_list)

    def setup_model(self, hparams):
        self.model = CharRNNGenerator(
            emb_size=hparams.emb_size,
            dropout=hparams.dropout,
            string_type=hparams.string_type,
            dataset=hparams.dataset_name
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

    def sample(self, num_samples):
        offset = 0
        results = []
        while offset < num_samples:
            cur_num_samples = min(num_samples - offset, self.hparams.sample_batch_size)
            offset += cur_num_samples

            self.model.eval()
            with torch.no_grad():
                sequences = self.model.decode(cur_num_samples, max_len=self.hparams.max_len, device=self.device)

            results.extend(untokenize(sequence, string_type=self.hparams.string_type, dataset=self.hparams.dataset_name) for sequence in sequences.tolist())

        disable_rdkit_log()

        canonicalize_cls = {
            "smiles": canonicalize,
            "spm_unigram": canonicalize,
            "spm_bpe": canonicalize,
            "bpe": canonicalize,
            "uni": canonicalize,
            "selfies": canonicalize_selfies,
            "deep_smiles": canonicalize_deep_smiles
        }.get(hparams.string_type)

        smiles_list = list(map(canonicalize_cls, results))
        results = list(map(canonicalize_cls, results))
        enable_rdkit_log()
        # decode to smiles for evaluation
        if self.hparams.string_type == "selfies":
            smiles_list = Parallel(n_jobs=8)(delayed(sf.decoder)(selfies) for selfies in smiles_list if selfies is not None)
            results = Parallel(n_jobs=8)(delayed(sf.decoder)(selfies) for selfies in results if selfies is not None)
            return smiles_list, results
        elif self.hparams.string_type in ['smiles', 'spm_unigram', 'spm_bpe', 'bpe', 'uni']:
            return smiles_list, results
        elif self.hparams.string_type == 'deep_smiles':
            converter = Converter(rings=True, branches=True)
            smiles_list = Parallel(n_jobs=8)(delayed(converter.decode)(deep_smiles) for deep_smiles in smiles_list if deep_smiles is not None)
            results = Parallel(n_jobs=8)(delayed(converter.decode)(deep_smiles) for deep_smiles in results if deep_smiles is not None)
            return smiles_list, results
        else:
            raise ValueError(f"Undefined string type {self.hparams.string_type}")     


    @staticmethod
    def add_args(parser):
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
        parser.add_argument("--lr", type=float, default=2e-4)
        
        #
        parser.add_argument("--max_len", type=int, default=250)
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=1)
        parser.add_argument("--num_samples", type=int, default=100)
        parser.add_argument("--sample_batch_size", type=int, default=1000)
        parser.add_argument("--eval_moses", action="store_true")

        # select string representation
        parser.add_argument("--string_type", type=str, default="selfies")
        parser.add_argument("--is_character", type=str, default='token')
        parser.add_argument("--is_pretokenized", type=str, default='wopre')

        return parser


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    SmilesCharGeneratorLightningModule.add_args(parser)
    
    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--tag", type=str, default="default")
    parser.add_argument("--group", type=str, default='char_rnn')

    hparams = parser.parse_args()
    model = SmilesCharGeneratorLightningModule(hparams)
    
    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
    )
    trainer.fit(model)