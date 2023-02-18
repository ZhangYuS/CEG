# CEG: A joint model for causal commonsense events enhanced story ending generation

## Introduction

This is the pytorch implementation of our paper "CEG: A joint model for causal commonsense events enhanced story ending generation"

## Requirements

```
python==3.7.13
torch==1.8.1+cu111
transformers==4.18.0
pytorch-lightning==1.6.3
```

## Preprocessing

Preprocessed datasets can be downloaded from [here](https://drive.google.com/drive/folders/1fE_a2I-UCtbzkt6i930T9b6UsufZiW9y?usp=sharing).

Unzip the file and move it to `data`.

## Usage

### Training 

The following command is an example to train the model on the trarining set and evaluate on the development set.

```bash
python3 main.py \
--data_dir data \
--mode SharedEncoderBart \
--model_name_or_path init_model/bart-with-two-decoder-base \
--learning_rate 4e-5 \
--warmup_ratio 0 \
--max_triple_num 44 \
--max_triple_length 140 \
--max_decoder_input_length 90 \
--max_target_length 50 \
--train_batch_size 4 \
--eval_batch_size 8 \
--num_workers 32 \
--max_epoch 8 \
--do_train \
--do_predict \
--one_crossattention
```

Init parameter of the model can be downloaded [here](https://drive.google.com/drive/folders/1fE_a2I-UCtbzkt6i930T9b6UsufZiW9y?usp=sharing)

### Test

```bash
python3 main.py \
--data_dir data \
--mode SharedEncoderBart \
--model_name_or_path init_model/bart-with-two-decoder-base \
--learning_rate 4e-5 \
--warmup_ratio 0 \
--max_triple_num 44 \
--max_triple_length 140 \
--max_decoder_input_length 90 \
--max_target_length 50 \
--train_batch_size 2 \
--eval_batch_size 2 \
--num_workers 12 \
--max_epoch 8 \
--do_predict \
--ckpt_path ${checkpoint_path}\
--one_crossattention
```

We provide our trained checkpoint and can be downloaded [here](https://drive.google.com/drive/folders/1fE_a2I-UCtbzkt6i930T9b6UsufZiW9y?usp=sharing)