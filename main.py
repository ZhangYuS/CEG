# Copyright 2021 Zhongyang Zhang
# Contact: mirakuruyoo@gmai.com
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

""" This main entrance of the whole project.

    Most of the code should not be changed, please directly
    add all the input arguments of your model's constructor
    and the dataset file's constructor. The MInterface and
    DInterface can be seen as transparent to all your args.
"""
import json
import pytorch_lightning as pl
from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import BartTokenizer
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from scripts.evaluation.evaluation import evaluate
from scripts.data.dataset import KgDataloader

from scripts.models.SharedEncoderBart import SharedEncoderBart

from scripts.data.StoryInferencedataset import StoryInferencedataset

from scripts.utils import remove_blank



EXPERIMENT_MODE = {
    'SharedEncoderBart': (SharedEncoderBart, StoryInferencedataset, BartTokenizer, BartTokenizer),
}



def main(parser, args):
    pl.seed_everything(args.seed)
    model_class, dataset_class, encoder_tokenizer_class, decoder_tokenizer_class = EXPERIMENT_MODE[args.mode]
    parser = model_class.add_model_specific_args(parser)
    parser = KgDataloader.add_dataloader_specific_args(parser)
    parser = dataset_class.add_dataset_specific_args(parser)
    args = parser.parse_args()
    args.encoder_tokenizer_class = encoder_tokenizer_class
    args.decoder_tokenizer_class = decoder_tokenizer_class
    args.dataset_class = dataset_class
    args.model_class = model_class
    if hasattr(args, "model_name_or_path"):
        encoder_tokenizer = encoder_tokenizer_class.from_pretrained(args.model_name_or_path)
        decoder_tokenizer = decoder_tokenizer_class.from_pretrained(args.model_name_or_path)
    else:
        encoder_tokenizer = encoder_tokenizer_class.from_pretrained(args.encoder_path)
        decoder_tokenizer = decoder_tokenizer_class.from_pretrained(args.decoder_path)

    if encoder_tokenizer.pad_token_id is None:
        encoder_tokenizer.add_special_tokens({'pad_token': encoder_tokenizer.decoder[0]})
    if decoder_tokenizer.pad_token is None:
        decoder_tokenizer.add_special_tokens({'pad_token': decoder_tokenizer.decoder[0]})

    args.encoder_tokenizer = encoder_tokenizer
    args.decoder_tokenizer = decoder_tokenizer

    model = model_class(args)
    data_loader = KgDataloader(**vars(args))
    logger = TensorBoardLogger(save_dir='logs', version=args.version_num)

    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(filename='{epoch}-{step}-{val_loss:.2f}-{val_B1:.4f}', save_top_k=args.save_top_k, monitor='val_B1', mode='max')

    trainer = Trainer.from_argparse_args(args, val_check_interval=0.25, callbacks=[checkpoint_callback, lr_monitor], accelerator="gpu", devices=1, logger=logger)
    if args.auto_lr_find:
        trainer.tune(model, data_loader)
    if args.do_train:
        trainer.fit(model, data_loader, ckpt_path=args.ckpt_path)
        if args.do_predict:
            result = trainer.predict(model, data_loader, ckpt_path='best')
    elif args.do_predict:
        # if args.ckpt_path is None:
        #     raise ValueError('no checkpoint path, use --ckpt_path')
        result = trainer.predict(model, data_loader, ckpt_path=args.ckpt_path)
    elif args.do_eval:
        if args.ckpt_path is None:
            raise ValueError('no checkpoint path, use --ckpt_path')
        trainer.validate(model, data_loader, ckpt_path=args.ckpt_path)
    else:
        raise ValueError


    if args.do_predict:
        # cross_attention = [i for x in result for i in x[1]]
        result = [i for x in result for i in x]
        for i in range(len(result)):
            result[i] = decoder_tokenizer.decode(result[i])
        checkpoint_path = checkpoint_callback.best_model_path if args.do_train else args.ckpt_path
        result = remove_blank(result)
        with open(checkpoint_path + '.pre', 'w', encoding='utf8') as f:
            for i in range(len(result)):
                print(result[i], file=f)

        gts = data_loader.ground_true()
        gts = remove_blank(gts)
        res = evaluate(result, gts)
        with open(checkpoint_path + '.res', 'w', encoding='utf8') as f:
            print(json.dumps(res), file=f)

        import torch
        # torch.save(cross_attention, checkpoint_path + '.crossattention')





def add_system_specific_args(parser: ArgumentParser):
    parser.add_argument('--seed', default=44, type=int)
    parser.add_argument('--do_train', action='store_true', help='weather train model')
    parser.add_argument('--do_predict', action='store_true', help='weather predict')
    parser.add_argument('--do_eval', action='store_true', help='weather predict')
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--version_num', type=int, required=True, help='the number of experiment')
    parser.add_argument('--mode', type=str, required=True, help='')
    parser.add_argument('--save_top_k', type=int, default=2)
    return parser


if __name__ == '__main__':
    parser = ArgumentParser()

    parser = Trainer.add_argparse_args(parser)
    parser = add_system_specific_args(parser)
    args = parser.parse_known_args()[0]
    main(parser, args)
