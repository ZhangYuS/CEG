import linecache
import random

import torch
import os
import json
from torch.utils.data import DataLoader, Dataset
from argparse import ArgumentParser

class StoryInferencedataset(Dataset):

    def __init__(self, decoder_tokenizer, encoder_tokenizer, file_path, max_triple_num, max_triple_length,
                 max_decoder_input_length, max_target_length, stage, **kwargs):

        self.stage = stage
        self.kwargs = kwargs
        self.max_decoder_input_length = max_decoder_input_length
        self.max_target_length = max_target_length
        self.max_triple_num = max_triple_num
        self.max_triple_length = max_triple_length
        self.data_length = None

        self.encoder_tokenizer = encoder_tokenizer
        self.decoder_tokenizer = decoder_tokenizer
        self.decoder_bos = self.decoder_tokenizer.bos_token_id if self.decoder_tokenizer.bos_token_id is not None else self.decoder_tokenizer.cls_token_id
        self.decoder_pad = self.decoder_tokenizer.pad_token_id
        self.decoder_eos = self.decoder_tokenizer.eos_token_id if self.decoder_tokenizer.eos_token_id is not None else self.decoder_tokenizer.sep_token_id
        self.encoder_cls = self.encoder_tokenizer.bos_token_id if self.encoder_tokenizer.bos_token_id is not None else self.encoder_tokenizer.cls_token_id
        self.encoder_pad = self.encoder_tokenizer.pad_token_id
        self.encoder_eos = self.encoder_tokenizer.eos_token_id if self.encoder_tokenizer.eos_token_id is not None else self.encoder_tokenizer.sep_token_id

        self.file_path = file_path

        self.relation_type_list = self.kwargs['relation_type_list']
        self.relation_type_list = [int(x) for x in self.relation_type_list]

        self.prompt_template = [
            'event that directly causes <sentence> is',
            'emotion that motivates <sentence> is',
            'location state that enables <sentence> is',
            'possess state that enables <sentence> is',
            'other attributes that enables <sentence> is',
            'event that <sentence> directly cause is',
            'emotion that caused by <sentence> is',
            'change in location that <sentence> results in is',
            'change of possession that <sentence> results in is',
            'other changes that <sentence> results in is',
        ]

        self.prompt_template = [self.prompt_template[x] for x in self.relation_type_list]

        self.position_offset = 2

    def __len__(self):
        if self.data_length is None:
            with open(self.file_path, 'r', encoding='utf8') as f:
                self.data_length = len(f.readlines())
        return self.data_length

    def __getitem__(self, idx):
        index = idx + 1
        data = linecache.getline(self.file_path, index).strip()
        data = json.loads(data)

        source = data['story']
        target = data['target']
        inference_sentence = data['origin_sentence']

        inference_sentence = [inference_sentence[idx] for idx in range(len(inference_sentence)) if idx % 10 in self.relation_type_list]

        source_input_ids = []
        for idx, s in enumerate(source):
            source_input_ids.extend(self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.tokenize(s)))
            source_input_ids.append(self.encoder_eos)

        inference_prompt = []
        for idx, sentence in enumerate(inference_sentence):
            template = self.prompt_template[idx % len(self.relation_type_list)]
            inference_story = source[idx // 10]
            inference_story = inference_story.strip()
            while not (inference_story[-1].isalpha() or inference_story[-1].isdigit()):
                inference_story = inference_story[:-1]
            inference_prompt.append(template.replace('<sentence>', inference_story))
        inference_prompt_ids = self.list_convert_ids(self.decoder_tokenizer, inference_prompt)
        inference_sentence_ids = self.list_convert_ids(self.decoder_tokenizer, [' ' + x for x in inference_sentence])

        encoder_input_ids = source_input_ids
        encoder_input_mask = [1 for _ in range(len(encoder_input_ids))]

        decoder_input_ids = [[self.decoder_eos, self.decoder_bos] + x for x in inference_prompt_ids]
        decoder_input_mask = [[1 for _ in range(len(x))] for x in decoder_input_ids]

        decoder_target_ids = [x + [self.decoder_eos] for x in inference_sentence_ids]
        decoder_target_mask = [1 for _ in range(len(decoder_target_ids))]

        second_decoder_input_ids = [self.decoder_eos, self.decoder_bos]
        # second_decoder_input_ids = [self.decoder_bos, self.decoder_eos]
        second_decoder_input_mask = [1 for _ in range(len(second_decoder_input_ids))]

        second_decoder_target_ids = self.decoder_tokenizer.convert_tokens_to_ids(self.decoder_tokenizer.tokenize(target)) + [self.decoder_eos]
        second_decoder_target_mask = [1 for _ in range(len(second_decoder_target_ids))]

        return {
            'encoder_input_ids': encoder_input_ids,
            'encoder_input_mask': encoder_input_mask,
            'decoder_input_ids': decoder_input_ids,
            'decoder_input_mask': decoder_input_mask,
            'second_decoder_input_ids': second_decoder_input_ids,
            'second_decoder_input_mask': second_decoder_input_mask,
            'decoder_target_ids': decoder_target_ids,
            'decoder_target_mask': decoder_target_mask,
            'second_decoder_target_ids': second_decoder_target_ids,
            'second_decoder_target_mask': second_decoder_target_mask
        }


    def collact_fn(self, batch):
        '''

        Args:
            batch:

        Returns:

        '''

        encoder_input_ids = [x['encoder_input_ids'] for x in batch]
        decoder_input_ids = [x['decoder_input_ids'] for x in batch]
        second_decoder_input_ids = [x['second_decoder_input_ids'] for x in batch]
        decoder_target_ids = [x['decoder_target_ids'] for x in batch]
        second_decoder_target_ids = [x['second_decoder_target_ids'] for x in batch]

        decoder_current_length = [[len(x) for x in y] for y in decoder_input_ids]
        second_decoder_current_length = [len(x) for x in second_decoder_input_ids]

        decoder_input_with_target_ids = [[decoder_input_ids[batch_idx][sample_idx] + decoder_target_ids[batch_idx][sample_idx] for sample_idx in range(len(decoder_input_ids[batch_idx]))] for batch_idx in range(len(decoder_input_ids))]
        decoder_label_ids = [[([-100 for _ in range(len(decoder_input_ids[batch_idx][sample_idx]) - 1)] + decoder_target_ids[batch_idx][sample_idx] + [self.decoder_eos]) for sample_idx in range(len(decoder_input_ids[batch_idx]))] for batch_idx in range(len(decoder_input_ids))]
        second_decoder_input_with_target_ids = [second_decoder_input_ids[idx] + second_decoder_target_ids[idx] for idx in range(len(second_decoder_input_ids))]
        second_decoder_label_ids = [[self.decoder_bos] + second_decoder_target_ids[idx] + [self.decoder_eos] for idx in range(len(second_decoder_target_ids))]

        encoder_input_ids, encoder_input_mask = self.pad_token(encoder_input_ids, self.encoder_pad)
        decoder_input_ids, decoder_input_mask = self.pad_token(decoder_input_ids, self.decoder_pad)
        second_decoder_input_ids, second_decoder_input_mask = self.pad_token(second_decoder_input_ids, self.decoder_pad)
        decoder_input_with_target_ids, decoder_input_with_target_mask = self.pad_token(decoder_input_with_target_ids, self.decoder_pad)
        decoder_label_ids, _ = self.pad_token(decoder_label_ids, -100)
        second_decoder_input_with_target_ids, second_decoder_input_with_target_mask = self.pad_token(second_decoder_input_with_target_ids, self.decoder_pad)
        second_decoder_label_ids, _ = self.pad_token(second_decoder_label_ids, -100)
        decoder_target_ids, decoder_target_mask = self.pad_token(decoder_target_ids, self.decoder_pad)
        second_decoder_target_ids, second_decoder_target_mask = self.pad_token(second_decoder_target_ids, self.decoder_pad)

        return_dict = {}
        return_dict['encoder_input_ids'] = encoder_input_ids
        return_dict['encoder_input_mask'] = encoder_input_mask
        return_dict['decoder_current_length'] = torch.LongTensor(decoder_current_length)
        return_dict['second_decoder_current_length'] = torch.LongTensor(second_decoder_current_length)
        return_dict['decoder_label_ids'] = decoder_label_ids
        return_dict['second_decoder_label_ids'] = second_decoder_label_ids
        return_dict['decoder_target_ids'] = decoder_target_ids
        return_dict['second_decoder_target_ids'] = second_decoder_target_ids
        return_dict['decoder_target_mask'] = decoder_target_mask
        return_dict['second_decoder_target_mask'] = second_decoder_target_mask

        if self.stage == 'test':
            return_dict['decoder_input_ids'] = decoder_input_ids
            return_dict['decoder_input_mask'] = decoder_input_mask
            return_dict['second_decoder_input_ids'] = second_decoder_input_ids
            return_dict['second_decoder_input_mask'] = second_decoder_input_mask

        else:
            return_dict['decoder_input_ids'] = decoder_input_with_target_ids
            return_dict['decoder_input_mask'] = decoder_input_with_target_mask
            return_dict['second_decoder_input_ids'] = second_decoder_input_with_target_ids
            return_dict['second_decoder_input_mask'] = second_decoder_input_with_target_mask

        return return_dict

    def list_convert_ids(self, tokenizer, l):
        res = []
        for s in l:
            res.append(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s)))
        return res

    def convert_ids(self, tokenizer, s):
        return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(s))

    def pad_token(self, id_list, pad_token, max_length=None, fix_max_length=False, suffix_truncate=True):
        if fix_max_length and max_length is None:
            raise ValueError('max_length can not be None if fix_max_length is True')

        if max_length is None:
            max_length = self.find_max_length(id_list)

        if not fix_max_length:
            max_length = min(self.find_max_length(id_list), max_length)

        if len(id_list[0]) != 0 and type(id_list[0][0]) == list:
            mask = []
            for i in range(len(id_list)):
                output = self.pad_token(id_list[i], pad_token, max_length=max_length, fix_max_length=True, suffix_truncate=suffix_truncate)
                id_list[i] = output[0]
                mask.append(output[1])
            return torch.stack(id_list), torch.stack(mask)
        else:
            for i in range(len(id_list)):
                if len(id_list[i]) <= max_length:
                    id_list[i] += [pad_token] * (max_length - len(id_list[i]))
                else:
                    if suffix_truncate:
                        id_list[i] = id_list[i][:max_length]
                    else:
                        id_list[i] = id_list[i][-max_length:]

            id_tensor = torch.LongTensor(id_list)
            mask = torch.where(id_tensor == pad_token, 0, 1)
            return id_tensor, mask

    def find_max_length(self, id_list):
        ret = 0
        if len(id_list[0]) != 0 and type(id_list[0][0]) == list:
            for i in range(len(id_list)):
                ret = max(ret, self.find_max_length(id_list[i]))
        else:
            ret = max([len(x) for x in id_list])
        return ret

    def pad_batch(self, list_batch, max_list_length, pad_token=None):
        len_list = [len(x) for x in list_batch]
        max_batch_len = min([max(len_list), max_list_length])
        if max_batch_len == 0:
            max_batch_len = 1
        len_list = [min(x, max_batch_len) for x in len_list]
        for i in range(len(list_batch)):
            if len(list_batch[i]) > max_batch_len:
                list_batch[i] = list_batch[i][:max_batch_len]
            elif len(list_batch[i]) < max_batch_len:
                list_batch[i] += [[pad_token] for _ in range(max_batch_len - len(list_batch[i]))]


        batch_mask = torch.Tensor([[1 for _ in range(len_list[i])] + [0 for _ in range(max_batch_len - len_list[i])] for i in range(len(list_batch))])
        return list_batch, batch_mask

    def ground_true(self):
        res = []
        with open(self.file_path, 'r', encoding='utf8') as f:
            for line in f.readlines():
                data = json.loads(line)
                if hasattr(self.decoder_tokenizer, 'do_lower_case') and self.decoder_tokenizer.do_lower_case:
                    res.append(data['target'].lower())
                else:
                    res.append(data['target'])
        return res

    @staticmethod
    def add_dataset_specific_args(parser: ArgumentParser):
        parser.add_argument(
            "--max_triple_num",
            default=None,
            type=int,
            required=True,
            help="Max number of the triple"
        )

        parser.add_argument(
            "--max_triple_length",
            default=None,
            type=int,
            required=True,
            help="Max_length of the triple"
        )

        parser.add_argument(
            "--max_decoder_input_length",
            default=None,
            type=int,
            required=True,
            help="Max length of decoder input"
        )

        parser.add_argument(
            "--max_target_length",
            default=None,
            type=int,
            required=True,
            help="Max length of target"
        )

        parser.add_argument("--data_label", type=str, help='threshold or hard', required=False, default=None)

        parser.add_argument("--relation_type_list", nargs='+', default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        return parser