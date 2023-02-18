# coding=utf-8
import json
import os

import torch
import torch.nn as nn
import pytorch_lightning as pl
import tqdm
from transformers import get_linear_schedule_with_warmup, BartConfig

from .transformer_Bart.SharedEncoderModel import SharedEncoderModel
from scripts.utils import remove_blank
from scripts.evaluation.evaluation import evaluate
from scripts.models.model_component.gate_model import GateModel


class SharedEncoderBart(pl.LightningModule):

    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.model_name_or_path = self.hparams.model_name_or_path
        self.second_decoder = self.hparams.second_decoder
        self.ff_decoder_input_id = self.hparams.ff_decoder_input_id
        self.fs_decoder_input_id = self.hparams.fs_decoder_input_id
        self.sf_decoder_input_id = self.hparams.sf_decoder_input_id
        self.ss_decoder_input_id = self.hparams.ss_decoder_input_id

        self.config = BartConfig.from_pretrained(self.model_name_or_path)
        self.bart_model: SharedEncoderModel = SharedEncoderModel.from_pretrained(self.model_name_or_path, config=self.config, args=self.hparams)

    def forward(self, encoder_input_ids=None,
                encoder_input_mask=None,
                decoder_input_ids=None,
                decoder_input_mask=None,
                second_decoder_input_ids=None,
                second_decoder_input_mask=None,
                decoder_label_ids=None,
                second_decoder_label_ids=None,
                decoder_target_ids=None,
                decoder_target_mask=None,
                second_decoder_target_ids=None,
                second_decoder_target_mask=None,
                decoder_current_length=None,
                second_decoder_current_length=None):

        output = self.bart_model(encoder_input_ids=encoder_input_ids,
                                 encoder_input_mask=encoder_input_mask,
                                 decoder_input_ids=decoder_input_ids,
                                 decoder_input_mask=decoder_input_mask,
                                 second_decoder_input_ids=second_decoder_input_ids,
                                 second_decoder_input_mask=second_decoder_input_mask,
                                 decoder_label_ids=decoder_label_ids,
                                 second_decoder_label_ids=second_decoder_label_ids,
                                 decoder_target_ids=decoder_target_ids,
                                 decoder_target_mask=decoder_target_mask,
                                 second_decoder_target_ids=second_decoder_target_ids,
                                 second_decoder_target_mask=second_decoder_target_mask,
                                 decoder_current_length=decoder_current_length,
                                 second_decoder_current_length=second_decoder_current_length,
                                 return_dict=True)

        log_keys = ['loss', 'first_loss', 'second_loss']
        log_dict = {}
        for k in log_keys:
            if k not in output.keys() or output[k] is None:
                continue
            log_dict[k] = output[k]
        return output['loss'], log_dict

    def training_step(self, batch, batch_idx):
        loss, log_dict = self(**batch)
        self.log_dict(log_dict)
        return loss

    def validation_step(self, batch, batch_idx):
        if self.trainer.sanity_checking and batch_idx == 0:
            if self.trainer.sanity_checking and batch_idx == 0:
                self.log_batch(batch, 'validation')

        loss, log_dict = self(**batch)
        return {'val_loss': loss}

    def validation_epoch_end(self, validation_step_outputs):
        val_loss = [x['val_loss'] for x in validation_step_outputs]
        val_loss = torch.mean(torch.Tensor(val_loss))

        result = []
        val_dataloader = self.trainer.datamodule.val_predict_dataloader()

        if self.trainer.sanity_checking:
            max_generate_idx = 3
        else:
            max_generate_idx = 10000000
        for batch_idx, batch in tqdm.tqdm(enumerate(val_dataloader), desc='predict in fit stage'):
            if batch_idx > max_generate_idx:
                break
            for k in batch.keys():
                if isinstance(batch[k], torch.Tensor):
                    batch[k] = batch[k].to(self.device)
            result.append(self.predict_step(batch, batch_idx))
        result = [i for x in result for i in x]
        for i in range(len(result)):
            result[i] = self.hparams.decoder_tokenizer.decode(result[i])
        result = remove_blank(result)

        epoch_num = self.trainer.current_epoch
        step_num = self.trainer.global_step
        if self.trainer.log_dir is not None:
            with open(os.path.join(self.trainer.log_dir, f'epoch_{epoch_num}_step_{step_num}.res'), 'w', encoding='utf8') as f:
                for l in result:
                    print(l, file=f)
        gts = self.trainer.datamodule.dev_ground_true()[:len(result)]
        gts = remove_blank(gts)
        res = evaluate(result, gts)
        test_B1 = res['B1']
        self.log('val_loss', val_loss)
        self.log('val_B1', test_B1)

    def test_step(self, batch, batch_idx):
        output_dict = self(**batch)
        self.log('test loss', output_dict['loss'])

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        if self.trainer.sanity_checking and batch_idx == 0:
            self.log_batch(batch, 'predict')

        result = self.bart_model.custom_generate(
            **batch,
            max_target_length=self.hparams.max_target_length,
            output_attentions=True,
            state=self.trainer.state.fn,
            log_dir=self.trainer.log_dir
        )
        result = result.tolist()
        for i in range(len(result)):
            try:
                idx = result[i].index(self.bart_model.config.bos_token_id)
                # idx = 1
            except ValueError:
                idx = -1
            result[i] = result[i][idx + 1:]
            try:
                idx = result[i].index(self.bart_model.config.eos_token_id)
            except ValueError:
                idx = len(result[i])
            result[i] = result[i][:idx]
        return result

    def setup(self, stage=None) -> None:
        if stage != "fit":
            return
        # Get dataloader by calling it - train_dataloader() is called after setup() by default
        train_loader = self.trainer.datamodule.train_dataloader()

        # Calculate total steps
        batch_size = self.hparams.train_batch_size
        accumulate_batch_size = self.trainer.accumulate_grad_batches * batch_size
        self.batch_steps = len(train_loader.dataset) // accumulate_batch_size
        self.total_steps = len(train_loader.dataset) * self.trainer.max_epochs // accumulate_batch_size

    def configure_optimizers(self):
        if hasattr(self.hparams, 'weight_decay'):
            weight_decay = self.hparams.weight_decay
        else:
            weight_decay = 0
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=int(self.batch_steps * self.hparams.warmup_ratio), num_training_steps=self.total_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def log_batch(self, batch, phase):
        tensorboard = self.logger.experiment

        tensorboard.add_text(f'{phase} batch keys', ' '.join(batch.keys()))
        for k in batch.keys():
            if not isinstance(batch[k], list):
                tensorboard.add_text(f'{phase} {k}', str(batch[k]))
            else:
                tensorboard.add_text(f'{phase} {k}', str(batch[k]))
            if k == 'input_ids':
                output_batch = batch[k].view(-1, batch[k].shape[-1])
                for i in range(output_batch.shape[0]):
                    tensorboard.add_text(f'{phase} token {k}[{i}]', self.hparams.encoder_tokenizer.decode(output_batch[i].tolist()))
            if k == 'decoder_input_ids':
                output_batch = batch[k].view(-1, batch[k].shape[-1])
                for i in range(output_batch.shape[0]):
                    tensorboard.add_text(f'{phase} token {k}[{i}]', self.hparams.encoder_tokenizer.decode(output_batch[i].tolist()))

    @staticmethod
    def add_model_specific_args(parser):
        parser.add_argument(
            "--model_name_or_path",
            default=None,
            type=str,
            required=True,
            help="Path to pretrained model or model identifier from huggingface.co/models",
        )

        parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")

        parser.add_argument(
            "--lr_scheduler",
            default="linear",
            type=str,
            help="Learning rate scheduler",
        )
        parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
        parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
        parser.add_argument("--warmup_ratio", default=0, type=float, help="Linear warmup over warmup_steps.")

        parser.add_argument("--adafactor", action="store_true")

        parser.add_argument("--second_decoder", action="store_true")
        parser.add_argument("--ff_decoder_input_id", type=int, default=None)
        parser.add_argument("--fs_decoder_input_id", type=int, default=None)
        parser.add_argument("--sf_decoder_input_id", type=int, default=None)
        parser.add_argument("--ss_decoder_input_id", type=int, default=None)
        parser.add_argument('--inference_type', type=str, default='all_hidden_states')
        parser.add_argument('--one_crossattention', action='store_true')
        parser.add_argument('--num_beam', type=int, default=-1)

        return parser