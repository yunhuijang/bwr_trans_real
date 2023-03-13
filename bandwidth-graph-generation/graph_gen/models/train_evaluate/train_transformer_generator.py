import argparse

import pytorch_lightning as pl
import wandb
from pytorch_lightning.loggers import WandbLogger

from transformer_generator import TransformerGenerator
from train_lstm_generator import LSTMGeneratorLightningModule
from graph_gen.models.model_utils import compute_sequence_accuracy, compute_sequence_cross_entropy

class TransformerGeneratorLightningModule(LSTMGeneratorLightningModule):
    def __init__(self, hparams):
        super().__init__(hparams)
    
    
    def setup_model(self, hparams):
        self.model = TransformerGenerator(
            num_layers=hparams.num_layers,
            emb_size=hparams.emb_size,
            nhead=hparams.nhead,
            dim_feedforward=hparams.dim_feedforward,
            input_dropout=hparams.input_dropout,
            dropout=hparams.dropout,
            vocab_size=self.vocab_size,
            max_len=self.max_len,
            pos_abs=hparams.pos_abs
        )
        
    def shared_step(self, batched_data):
        loss, statistics = 0.0, dict()
        logits = self.model(batched_data)
        loss = compute_sequence_cross_entropy(logits, batched_data[1], ignore_index=0)
        statistics["loss/total"] = loss
        statistics["acc/total"] = compute_sequence_accuracy(logits, batched_data[1], ignore_index=0)[0]
        
        return loss, statistics
    
    @staticmethod
    def add_args(parser):
        parser.add_argument("--dataset_name", type=str, default="ENZYMES")
        parser.add_argument("--batch_size", type=int, default=128)
        parser.add_argument("--num_workers", type=int, default=6)
        parser.add_argument("--order", type=str, default='C-M')
        parser.add_argument("--replicate", type=int, default=0)

        #
        parser.add_argument("--model", type=str, default='transformer')
        parser.add_argument("--num_layers", type=int, default=3)
        parser.add_argument("--emb_size", type=int, default=512)
        parser.add_argument("--dim_feedforward", type=int, default=2048)
        parser.add_argument("--input_dropout", type=int, default=0.0)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--nhead", type=int, default=8)
        parser.add_argument("--logit_hidden_dim", type=int, default=256)

        #
        parser.add_argument("--lr", type=float, default=2e-4)
        parser.add_argument("--gradient_clip_val", type=int, default=1)
        
        #
        parser.add_argument("--num_samples", type=int, default=100)
        parser.add_argument("--sample_batch_size", type=int, default=1000)
        
        parser.add_argument("--check_sample_every_n_epoch", type=int, default=1)
        parser.add_argument("--max_epochs", type=int, default=10)
        parser.add_argument("--group", type=str, default='transformer')
        
        parser.add_argument("--wandb_on", type=str, default='disabled')
        
        # transformer
        parser.add_argument("--pos_abs", type=bool, default=True)

        return parser


if __name__ == "__main__":
    # wandb.init(name='QM9-SMILES_Transformer')
    parser = argparse.ArgumentParser()
    TransformerGeneratorLightningModule.add_args(parser)

    hparams = parser.parse_args()

    wandb_logger = WandbLogger(name=f'{hparams.dataset_name}-{hparams.model}', 
                               project='bwr', group=f'{hparams.group}', mode=f'{hparams.wandb_on}')
    
    wandb.config.update(hparams)

    model = TransformerGeneratorLightningModule(hparams)
    
    trainer = pl.Trainer(
        gpus=1,
        default_root_dir="../resource/log/",
        max_epochs=hparams.max_epochs,
        logger=wandb_logger
    )
    trainer.fit(model)