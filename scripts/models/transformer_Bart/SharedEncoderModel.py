import copy
import math
import os
import random
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from scripts.utils import autoregress_generation

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)

from transformers.utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)
from transformers.models.bart.configuration_bart import BartConfig
from transformers.models.bart.modeling_bart import (
    BartAttention,
    BartConfig,
    BartEncoderLayer,
    BartDecoderLayer,
    _make_causal_mask,
    _expand_mask,
    BART_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    _EXPECTED_OUTPUT_SHAPE,
    shift_tokens_right,
    BART_START_DOCSTRING,
    BART_GENERATION_EXAMPLE,
    logger
)

from .modeling_utils import PreTrainedModel


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids_shape: torch.Size, past_key_values_length: int = 0):
        """`input_ids_shape` is expected to be [bsz x seqlen]."""
        bsz, seq_len = input_ids_shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len, dtype=torch.long, device=self.weight.device
        )
        return super().forward(positions + self.offset)

    def forward_position_ids(self, position_ids: torch.LongTensor):
        bsz, seq_len = position_ids.shape[:2]
        return super().forward(position_ids + self.offset)


class BartPretrainedModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version"]

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (BartWithTwoCroDecoder, BartEncoder)):
            module.gradient_checkpointing = value

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class BartWithTwoCroDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.second_encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.second_encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        second_encoder_hidden_states: Optional[torch.Tensor] = None,
        second_encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        second_cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape `(batch, seq_len, embed_dim)`
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                `(encoder_attention_heads,)`.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size `(decoder_attention_heads,)`.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[2: 4] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=output_attentions,
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        second_cross_attn_present_key_value = None
        second_cross_attn_weights = None
        if second_encoder_hidden_states is not None:
            residual = hidden_states

            second_cross_attn_past_key_value = past_key_value[4: 6] if past_key_value is not None else None
            hidden_states, second_cross_attn_weights, second_cross_attn_present_key_value = self.second_encoder_attn(
                hidden_states=hidden_states,
                key_value_states=second_encoder_hidden_states,
                attention_mask=second_encoder_attention_mask,
                layer_head_mask=second_cross_attn_layer_head_mask,
                past_key_value=second_cross_attn_past_key_value,
                output_attentions=output_attentions
            )
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.second_encoder_attn_layer_norm(hidden_states)
            present_key_value = present_key_value + second_cross_attn_present_key_value


        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights, second_cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


class BartEncoder(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        embed_pos = self.embed_positions(input_shape)

        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # expand attention_mask
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        # check if head_mask has a correct number of layers specified if desired
        if head_mask is not None:
            if head_mask.size()[0] != (len(self.layers)):
                raise ValueError(
                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                )

        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = (None, None)
            else:
                if self.gradient_checkpointing and self.training:

                    def create_custom_forward(module):
                        def custom_forward(*inputs):
                            return module(*inputs, output_attentions)

                        return custom_forward

                    layer_outputs = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(encoder_layer),
                        hidden_states,
                        attention_mask,
                        (head_mask[idx] if head_mask is not None else None),
                    )
                else:
                    layer_outputs = encoder_layer(
                        hidden_states,
                        attention_mask,
                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                        output_attentions=output_attentions,
                    )

                hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class BartWithTwoCroDecoder(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartWithTwoCroDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        position_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        second_encoder_hidden_states: Optional[torch.FloatTensor] = None,
        second_encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        second_cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch_size, encoder_sequence_length, hidden_size)`, *optional*):
                Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
                of the decoder.
            encoder_attention_mask (`torch.LongTensor` of shape `(batch_size, encoder_sequence_length)`, *optional*):
                Mask to avoid performing cross-attention on padding tokens indices of encoder input_ids. Mask values
                selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            cross_attn_head_mask (`torch.Tensor` of shape `(decoder_layers, decoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the cross-attention modules in the decoder to avoid performing
                cross-attention on hidden heads. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
                Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
                shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`) and 2 additional tensors of
                shape `(batch_size, num_heads, encoder_sequence_length, embed_size_per_head)`.

                Contains pre-computed hidden-states (key and values in the self-attention blocks and in the
                cross-attention blocks) that can be used (see `past_key_values` input) to speed up sequential decoding.

                If `past_key_values` are used, the user can optionally input only the last `decoder_input_ids` (those
                that don't have their past key value states given to this model) of shape `(batch_size, 1)` instead of
                all `decoder_input_ids` of shape `(batch_size, sequence_length)`. inputs_embeds (`torch.FloatTensor` of
                shape `(batch_size, sequence_length, hidden_size)`, *optional*): Optionally, instead of passing
                `input_ids` you can choose to directly pass an embedded representation. This is useful if you want more
                control over how to convert `input_ids` indices into associated vectors than the model's internal
                embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        if second_encoder_hidden_states is not None and second_encoder_attention_mask is not None:
            second_encoder_attention_mask = _expand_mask(second_encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        if position_ids is None:
            positions = self.embed_positions(input_shape, past_key_values_length)
        else:
            positions = self.embed_positions.forward_position_ids(position_ids)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
                raise NotImplementedError
            else:

                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    second_encoder_hidden_states=second_encoder_hidden_states,
                    second_encoder_attention_mask=second_encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    second_cross_attn_layer_head_mask=(
                        second_cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[-1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1], )

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],) if layer_outputs[3] is not None else (layer_outputs[2], layer_outputs[3], )

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )


class BartModel(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartWithTwoCroDecoder(config, self.shared)
        self.second_decoder = BartWithTwoCroDecoder(config, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared
        self.second_decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def get_second_decoder(self):
        return self.second_decoder

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=Seq2SeqModelOutput,
        config_class=_CONFIG_FOR_DOC,
        expected_output=_EXPECTED_OUTPUT_SHAPE,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        raise NotImplementedError
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def encode(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        return encoder_outputs

    def decode(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[List[torch.FloatTensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        second_encoder_hidden_states: Optional[List[torch.FloatTensor]] = None,
        second_encoder_attention_mask: Optional[List[torch.FloatTensor]] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            second_encoder_hidden_states=second_encoder_hidden_states,
            second_encoder_attention_mask=second_encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

    def second_decode(
        self,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        encoder_hidden_states: Optional[List[torch.FloatTensor]] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        second_encoder_hidden_states: Optional[List[torch.FloatTensor]] = None,
        second_encoder_attention_mask: Optional[List[torch.FloatTensor]] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqModelOutput]:
        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.second_decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            second_encoder_hidden_states=second_encoder_hidden_states,
            second_encoder_attention_mask=second_encoder_attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions
        )

@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.", BART_START_DOCSTRING
)
class SharedEncoderModel(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, args=None):
        super().__init__(config)
        self.model = BartModel(config)

        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)
        self.register_buffer("second_final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.second_lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        self.args = args

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def _tie_weights(self):
        pass

    @add_start_docstrings_to_model_forward(BART_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_CONFIG_FOR_DOC)
    @add_end_docstrings(BART_GENERATION_EXAMPLE)
    def forward(
        self,
        encoder_input_ids=None,
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
        second_decoder_current_length=None,
        past_key_values: Optional[List[List[torch.FloatTensor]]] = None,
        second_past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        generate=False,
        input_ids=None,
        input_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """

        if generate:
            outputs, hidden_states, lm_logits, lm_pro = self.second_decode(
                input_ids,
                input_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )
            return Seq2SeqLMOutput(
                logits=lm_logits,
                past_key_values=outputs.past_key_values,
                decoder_hidden_states=outputs.decoder_hidden_states,
                decoder_attentions=outputs.decoder_attentions,
                cross_attentions=outputs.cross_attentions,
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if len(encoder_input_ids.shape) == 3:
            batch_size, knowledge_num, seq_len = encoder_input_ids.shape
        else:
            knowledge_num = 1
            batch_size, seq_len = encoder_input_ids.shape

        encoder_input_ids = encoder_input_ids.view(batch_size * knowledge_num, seq_len)
        encoder_input_mask = encoder_input_mask.view(batch_size * knowledge_num, seq_len)
        encoder_outputs = self.model.encode(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state.view(batch_size, knowledge_num * seq_len, -1)
        encoder_hidden_states_mask = encoder_input_mask.view(batch_size, knowledge_num * seq_len)

        second_decoder_first_encoder_hidden_states = encoder_hidden_states
        second_decoder_first_encoder_hidden_states_mask = encoder_hidden_states_mask

        second_decoder_second_encoder_hidden_states = None
        second_decoder_second_encoder_hidden_states_mask = None

        if self.args.inference_type == 'all_hidden_states':
            if len(decoder_input_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_input_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_input_ids.shape

            decoder_input_ids = decoder_input_ids.view(batch_size * knowledge_num, seq_len)
            decoder_input_mask = decoder_input_mask.view(batch_size * knowledge_num, seq_len)

            first_decoder_encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, knowledge_num, 1, 1)
            first_decoder_encoder_hidden_states_mask = encoder_hidden_states_mask.unsqueeze(1).repeat(1, knowledge_num, 1)
            first_decoder_encoder_hidden_states = first_decoder_encoder_hidden_states.view(batch_size * knowledge_num, encoder_hidden_states.shape[-2], -1)
            first_decoder_encoder_hidden_states_mask = first_decoder_encoder_hidden_states_mask.view(batch_size * knowledge_num, encoder_hidden_states.shape[-2])

            first_decoder_output, decoder_hidden_states, lm_logits, first_lm_pro = self.decode(
                decoder_input_ids,
                decoder_input_mask,
                encoder_hidden_states=first_decoder_encoder_hidden_states,
                encoder_attention_mask=first_decoder_encoder_hidden_states_mask,
                second_encoder_hidden_states=None,
                second_encoder_attention_mask=None,
                past_key_values=past_key_values[0] if past_key_values is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )


            decoder_hidden_states_mask = decoder_input_mask
            decoder_encoder_output = self.model.encode(
                inputs_embeds=decoder_hidden_states,
                attention_mask=decoder_hidden_states_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size, knowledge_num * seq_len, -1)
            second_decoder_second_encoder_hidden_states_mask = decoder_input_mask.view(batch_size, knowledge_num * seq_len)

        elif self.args.inference_type == 'generate_hidden_states':
            if len(decoder_input_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_input_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_input_ids.shape

            decoder_input_ids = decoder_input_ids.view(batch_size * knowledge_num, seq_len)
            decoder_input_mask = decoder_input_mask.view(batch_size * knowledge_num, seq_len)

            first_decoder_encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, knowledge_num, 1, 1)
            first_decoder_encoder_hidden_states_mask = encoder_hidden_states_mask.unsqueeze(1).repeat(1, knowledge_num,
                                                                                                      1)
            first_decoder_encoder_hidden_states = first_decoder_encoder_hidden_states.view(batch_size * knowledge_num,
                                                                                           encoder_hidden_states.shape[
                                                                                               -2], -1)
            first_decoder_encoder_hidden_states_mask = first_decoder_encoder_hidden_states_mask.view(
                batch_size * knowledge_num, encoder_hidden_states.shape[-2])

            first_decoder_output, decoder_hidden_states, lm_logits, first_lm_pro = self.decode(
                decoder_input_ids,
                decoder_input_mask,
                encoder_hidden_states=first_decoder_encoder_hidden_states,
                encoder_attention_mask=first_decoder_encoder_hidden_states_mask,
                second_encoder_hidden_states=None,
                second_encoder_attention_mask=None,
                past_key_values=past_key_values[0] if past_key_values is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            decoder_hidden_states_mask = torch.where(decoder_label_ids.view(batch_size * knowledge_num, seq_len) == -100, 0, torch.ones_like(decoder_label_ids.view(batch_size * knowledge_num, seq_len)))
            decoder_encoder_output = self.model.encode(
                inputs_embeds=decoder_hidden_states,
                attention_mask=decoder_hidden_states_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size,
                                                                                                        knowledge_num * seq_len,
                                                                                                        -1)
            second_decoder_second_encoder_hidden_states_mask = decoder_hidden_states_mask.view(batch_size,
                                                                                       knowledge_num * seq_len)


        elif self.args.inference_type == 'inference_ids':

            if len(decoder_input_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_input_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_input_ids.shape

            decoder_input_ids = decoder_input_ids.view(batch_size * knowledge_num, seq_len)
            decoder_input_mask = decoder_input_mask.view(batch_size * knowledge_num, seq_len)

            first_decoder_encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, knowledge_num, 1, 1)
            first_decoder_encoder_hidden_states_mask = encoder_hidden_states_mask.unsqueeze(1).repeat(1, knowledge_num,
                                                                                                      1)
            first_decoder_encoder_hidden_states = first_decoder_encoder_hidden_states.view(batch_size * knowledge_num,
                                                                                           encoder_hidden_states.shape[
                                                                                               -2], -1)
            first_decoder_encoder_hidden_states_mask = first_decoder_encoder_hidden_states_mask.view(
                batch_size * knowledge_num, encoder_hidden_states.shape[-2])

            first_decoder_output, decoder_hidden_states, lm_logits, first_lm_pro = self.decode(
                decoder_input_ids,
                decoder_input_mask,
                encoder_hidden_states=first_decoder_encoder_hidden_states,
                encoder_attention_mask=first_decoder_encoder_hidden_states_mask,
                second_encoder_hidden_states=None,
                second_encoder_attention_mask=None,
                past_key_values=past_key_values[0] if past_key_values is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            if len(decoder_target_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_target_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_target_ids.shape

            decoder_target_ids = decoder_target_ids.view(batch_size * knowledge_num, seq_len)
            decoder_target_mask = decoder_target_mask.view(batch_size * knowledge_num, seq_len)

            decoder_encoder_output = self.model.encode(
                input_ids=decoder_target_ids,
                attention_mask=decoder_target_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size,
                                                                                                        knowledge_num * seq_len,
                                                                                                        -1)
            second_decoder_second_encoder_hidden_states_mask = decoder_target_mask.view(batch_size,
                                                                                        knowledge_num * seq_len)

        elif self.args.inference_type == 'all_inference_ids':

            if len(decoder_input_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_input_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_input_ids.shape

            decoder_input_ids = decoder_input_ids.view(batch_size * knowledge_num, seq_len)
            decoder_input_mask = decoder_input_mask.view(batch_size * knowledge_num, seq_len)

            first_decoder_encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, knowledge_num, 1, 1)
            first_decoder_encoder_hidden_states_mask = encoder_hidden_states_mask.unsqueeze(1).repeat(1, knowledge_num,
                                                                                                      1)
            first_decoder_encoder_hidden_states = first_decoder_encoder_hidden_states.view(batch_size * knowledge_num,
                                                                                           encoder_hidden_states.shape[
                                                                                               -2], -1)
            first_decoder_encoder_hidden_states_mask = first_decoder_encoder_hidden_states_mask.view(
                batch_size * knowledge_num, encoder_hidden_states.shape[-2])

            first_decoder_output, decoder_hidden_states, lm_logits, first_lm_pro = self.decode(
                decoder_input_ids,
                decoder_input_mask,
                encoder_hidden_states=first_decoder_encoder_hidden_states,
                encoder_attention_mask=first_decoder_encoder_hidden_states_mask,
                second_encoder_hidden_states=None,
                second_encoder_attention_mask=None,
                past_key_values=past_key_values[0] if past_key_values is not None else None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            decoder_target_with_prompt_ids = decoder_input_ids[:, 1:]
            decoder_target_with_prompt_mask = decoder_input_mask[:, 1:]
            if len(decoder_target_with_prompt_ids.shape) == 3:
                _, _, seq_len = decoder_target_with_prompt_ids.shape
            else:
                _ = 1
                _, seq_len = decoder_target_with_prompt_ids.shape

            decoder_target_with_prompt_ids = decoder_target_with_prompt_ids.view(batch_size * knowledge_num, seq_len)
            decoder_target_with_prompt_mask = decoder_target_with_prompt_mask.contiguous().view(batch_size * knowledge_num, seq_len)

            decoder_encoder_output = self.model.encode(
                input_ids=decoder_target_with_prompt_ids,
                attention_mask=decoder_target_with_prompt_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size,
                                                                                                        knowledge_num * seq_len,
                                                                                                        -1)
            second_decoder_second_encoder_hidden_states_mask = decoder_target_with_prompt_mask.view(batch_size, knowledge_num * seq_len)

        elif self.args.inference_type == 'ground_truth_ids':
            if len(decoder_target_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_target_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_target_ids.shape

            decoder_target_ids = decoder_target_ids.view(batch_size * knowledge_num, seq_len)
            decoder_target_mask = decoder_target_mask.view(batch_size * knowledge_num, seq_len)

            decoder_encoder_output = self.model.encode(
                input_ids=decoder_target_ids,
                attention_mask=decoder_target_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size,
                                                                                                        knowledge_num * seq_len,
                                                                                                        -1)
            second_decoder_second_encoder_hidden_states_mask = decoder_target_mask.view(batch_size,
                                                                                               knowledge_num * seq_len)

        elif self.args.inference_type == 'no_inference':
            second_decoder_second_encoder_hidden_states = None
            second_decoder_second_encoder_hidden_states_mask = None

        else:
            raise ValueError

        if self.args.one_crossattention and second_decoder_second_encoder_hidden_states is not None:
            second_decoder_first_encoder_hidden_states = torch.cat([second_decoder_first_encoder_hidden_states, second_decoder_second_encoder_hidden_states], dim=1)
            second_decoder_first_encoder_hidden_states_mask = torch.cat([second_decoder_first_encoder_hidden_states_mask, second_decoder_second_encoder_hidden_states_mask], dim=1)
            second_decoder_second_encoder_hidden_states = None
            second_decoder_second_encoder_hidden_states_mask = None



        second_decoder_output, second_decoder_hidden_states, second_lm_logits, second_lm_pro = self.second_decode(
            second_decoder_input_ids,
            second_decoder_input_mask,
            encoder_hidden_states=second_decoder_first_encoder_hidden_states,
            encoder_attention_mask=second_decoder_first_encoder_hidden_states_mask,
            second_encoder_hidden_states=second_decoder_second_encoder_hidden_states,
            second_encoder_attention_mask=second_decoder_second_encoder_hidden_states_mask,
            past_key_values=second_past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        loss_fct = nn.NLLLoss()
        second_loss = loss_fct(second_lm_pro.view(-1, self.config.vocab_size), second_decoder_label_ids.view(-1))

        loss = second_loss

        if self.args.inference_type == 'all_hidden_states' or self.args.inference_type == 'generate_hidden_states' or self.args.inference_type=='inference_ids' or self.args.inference_type == 'all_inference_ids':
            first_loss = loss_fct(first_lm_pro.view(-1, self.config.vocab_size), decoder_label_ids.view(-1))
            loss += first_loss

        result_dict = {}

        result_dict['loss'] = loss
        result_dict['second_past_key_values'] = second_decoder_output.past_key_values
        result_dict['second_logits'] = second_lm_logits
        result_dict['second_loss'] = second_loss
        result_dict['second_decoder_hidden_states'] = second_decoder_output.last_hidden_state
        result_dict['second_decoder_attention'] = second_decoder_output.decoder_attentions
        result_dict['second_decoder_cross_attention'] = second_decoder_output.cross_attentions


        if self.args.inference_type == 'all_hidden_states' or self.args.inference_type == 'generate_hidden_states' or self.args.inference_type=='inference_ids' or self.args.inference_type == 'all_inference_ids':
            result_dict['past_key_values'] = first_decoder_output.past_key_values
            result_dict['first_logits'] = lm_logits
            result_dict['first_loss'] = first_loss
            result_dict['first_decoder_hidden_states'] = first_decoder_output.last_hidden_state
            result_dict['first_decoder_attention'] = first_decoder_output.decoder_attentions
            result_dict['first_decoder_cross_attention'] = first_decoder_output.cross_attentions

        return result_dict

    def decode(
        self,
        input_ids,
        input_mask,
        position_ids=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        second_encoder_hidden_states=None,
        second_encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):

        first_decoder_output = self.model.decode(
            decoder_input_ids=input_ids,
            decoder_attention_mask=input_mask,
            position_ids=position_ids,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            second_encoder_hidden_states=second_encoder_hidden_states,
            second_encoder_attention_mask=second_encoder_attention_mask,
            past_key_values=past_key_values if past_key_values is not None else None,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        decoder_hidden_states = first_decoder_output.last_hidden_state
        lm_logits = self.lm_head(decoder_hidden_states) + self.final_logits_bias
        first_lm_pro = nn.LogSoftmax(dim=-1)(lm_logits)

        return first_decoder_output, decoder_hidden_states, lm_logits, first_lm_pro

    def second_decode(
        self,
        input_ids,
        input_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        second_encoder_hidden_states=None,
        second_encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs
    ):

        second_decoder_output = self.model.second_decode(
            decoder_input_ids=input_ids,
            decoder_attention_mask=input_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            second_encoder_hidden_states=second_encoder_hidden_states,
            second_encoder_attention_mask=second_encoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        decoder_hidden_states = second_decoder_output.last_hidden_state
        second_lm_logits = self.second_lm_head(decoder_hidden_states) + self.second_final_logits_bias
        second_lm_pro = nn.LogSoftmax(dim=-1)(second_lm_logits)

        return second_decoder_output, decoder_hidden_states, second_lm_logits, second_lm_pro

    def custom_generate(
        self,
        encoder_input_ids=None,
        encoder_input_mask=None,
        decoder_input_ids=None,
        decoder_input_mask=None,
        second_decoder_input_ids=None,
        second_decoder_input_mask=None,
        decoder_current_length=None,
        second_decoder_current_length=None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        max_target_length=-1,
        decoder_target_ids=None,
        decoder_target_mask=None,
        second_decoder_target_ids=None,
        second_decoder_target_mask=None,
        decoder_label_ids=None,
        second_decoder_label_ids=None,
        **kwargs
    ) -> Union[Tuple, Seq2SeqLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        """

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if decoder_input_ids is None:
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        if len(encoder_input_ids.shape) == 3:
            batch_size, knowledge_num, seq_len = encoder_input_ids.shape
        else:
            knowledge_num = 1
            batch_size, seq_len = encoder_input_ids.shape

        encoder_input_ids = encoder_input_ids.view(batch_size * knowledge_num, seq_len)
        encoder_input_mask = encoder_input_mask.view(batch_size * knowledge_num, seq_len)
        encoder_outputs = self.model.encode(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        encoder_hidden_states = encoder_outputs.last_hidden_state.view(batch_size, knowledge_num * seq_len, -1)
        encoder_hidden_states_mask = encoder_input_mask.view(batch_size, knowledge_num * seq_len)

        second_decoder_first_encoder_hidden_states = encoder_hidden_states
        second_decoder_first_encoder_hidden_states_mask = encoder_hidden_states_mask

        second_decoder_second_encoder_hidden_states = None
        second_decoder_second_encoder_hidden_states_mask = None

        if self.args.inference_type == 'all_hidden_states':

            if len(decoder_input_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_input_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_input_ids.shape

            decoder_input_ids = decoder_input_ids.view(batch_size * knowledge_num, seq_len)
            decoder_input_mask = decoder_input_mask.view(batch_size * knowledge_num, seq_len)

            first_decoder_encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, knowledge_num, 1, 1)
            first_decoder_encoder_hidden_states_mask = encoder_hidden_states_mask.unsqueeze(1).repeat(1, knowledge_num, 1)
            first_decoder_encoder_hidden_states = first_decoder_encoder_hidden_states.view(batch_size * knowledge_num, encoder_hidden_states.shape[-2], -1)
            first_decoder_encoder_hidden_states_mask = first_decoder_encoder_hidden_states_mask.view(batch_size * knowledge_num, encoder_hidden_states.shape[-2])

            first_decoder_generate_output = autoregress_generation(
                self.decode,
                decoder_input_ids,
                decoder_input_mask,
                decoder_current_length,
                max_target_length,
                self.config.eos_token_id,
                encoder_hidden_states=first_decoder_encoder_hidden_states,
                encoder_attention_mask=first_decoder_encoder_hidden_states_mask,
                second_encoder_hidden_states=None,
                second_encoder_attention_mask=None,
                past_key_values=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )



            decoder_hidden_states_mask = first_decoder_generate_output['total_hidden_states_mask']
            decoder_hidden_states = first_decoder_generate_output['total_hidden_states']
            decoder_encoder_output = self.model.encode(
                inputs_embeds=decoder_hidden_states,
                attention_mask=decoder_hidden_states_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size, -1, self.base_model.config.d_model)
            second_decoder_second_encoder_hidden_states_mask = decoder_hidden_states_mask.view(batch_size, -1)

        if self.args.inference_type == 'generate_hidden_states':

            if len(decoder_input_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_input_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_input_ids.shape

            decoder_input_ids = decoder_input_ids.view(batch_size * knowledge_num, seq_len)
            decoder_input_mask = decoder_input_mask.view(batch_size * knowledge_num, seq_len)

            first_decoder_encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, knowledge_num, 1, 1)
            first_decoder_encoder_hidden_states_mask = encoder_hidden_states_mask.unsqueeze(1).repeat(1, knowledge_num,
                                                                                                      1)
            first_decoder_encoder_hidden_states = first_decoder_encoder_hidden_states.view(batch_size * knowledge_num,
                                                                                           encoder_hidden_states.shape[
                                                                                               -2], -1)
            first_decoder_encoder_hidden_states_mask = first_decoder_encoder_hidden_states_mask.view(
                batch_size * knowledge_num, encoder_hidden_states.shape[-2])

            first_decoder_generate_output = autoregress_generation(
                self.decode,
                decoder_input_ids,
                decoder_input_mask,
                decoder_current_length,
                max_target_length,
                self.config.eos_token_id,
                encoder_hidden_states=first_decoder_encoder_hidden_states,
                encoder_attention_mask=first_decoder_encoder_hidden_states_mask,
                second_encoder_hidden_states=None,
                second_encoder_attention_mask=None,
                past_key_values=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            decoder_hidden_states_mask = first_decoder_generate_output['generate_hidden_states_mask']
            decoder_hidden_states = first_decoder_generate_output['generate_hidden_states']
            decoder_encoder_output = self.model.encode(
                inputs_embeds=decoder_hidden_states,
                attention_mask=decoder_hidden_states_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size, -1,
                                                                                                        self.base_model.config.d_model)
            second_decoder_second_encoder_hidden_states_mask = decoder_hidden_states_mask.view(batch_size, -1)

        if self.args.inference_type == 'all_inference_ids':

            if len(decoder_input_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_input_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_input_ids.shape

            decoder_input_ids = decoder_input_ids.view(batch_size * knowledge_num, seq_len)
            decoder_input_mask = decoder_input_mask.view(batch_size * knowledge_num, seq_len)

            first_decoder_encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, knowledge_num, 1, 1)
            first_decoder_encoder_hidden_states_mask = encoder_hidden_states_mask.unsqueeze(1).repeat(1, knowledge_num, 1)
            first_decoder_encoder_hidden_states = first_decoder_encoder_hidden_states.view(batch_size * knowledge_num, encoder_hidden_states.shape[-2], -1)
            first_decoder_encoder_hidden_states_mask = first_decoder_encoder_hidden_states_mask.view(batch_size * knowledge_num, encoder_hidden_states.shape[-2])

            first_decoder_generate_output = autoregress_generation(
                self.decode,
                decoder_input_ids,
                decoder_input_mask,
                decoder_current_length,
                max_target_length,
                self.config.eos_token_id,
                encoder_hidden_states=first_decoder_encoder_hidden_states,
                encoder_attention_mask=first_decoder_encoder_hidden_states_mask,
                second_encoder_hidden_states=None,
                second_encoder_attention_mask=None,
                past_key_values=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            # if kwargs['state'] != 'fit':
            #     self.save_inference_result(first_decoder_generate_output, decoder_label_ids, batch_size, knowledge_num, kwargs['log_dir'])


            all_inference_ids = first_decoder_generate_output['total_result']
            all_inference_mask = first_decoder_generate_output['total_hidden_states_mask']
            decoder_encoder_output = self.model.encode(
                input_ids=all_inference_ids,
                attention_mask=all_inference_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size, -1, self.base_model.config.d_model)
            second_decoder_second_encoder_hidden_states_mask = all_inference_mask.view(batch_size, -1)


        elif self.args.inference_type == 'inference_ids':

            if len(decoder_input_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_input_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_input_ids.shape

            decoder_input_ids = decoder_input_ids.view(batch_size * knowledge_num, seq_len)
            decoder_input_mask = decoder_input_mask.view(batch_size * knowledge_num, seq_len)

            first_decoder_encoder_hidden_states = encoder_hidden_states.unsqueeze(1).repeat(1, knowledge_num, 1, 1)
            first_decoder_encoder_hidden_states_mask = encoder_hidden_states_mask.unsqueeze(1).repeat(1, knowledge_num,
                                                                                                      1)
            first_decoder_encoder_hidden_states = first_decoder_encoder_hidden_states.view(batch_size * knowledge_num,
                                                                                           encoder_hidden_states.shape[
                                                                                               -2], -1)
            first_decoder_encoder_hidden_states_mask = first_decoder_encoder_hidden_states_mask.view(
                batch_size * knowledge_num, encoder_hidden_states.shape[-2])

            first_decoder_generate_output = autoregress_generation(
                self.decode,
                decoder_input_ids,
                decoder_input_mask,
                decoder_current_length,
                max_target_length,
                self.config.eos_token_id,
                encoder_hidden_states=first_decoder_encoder_hidden_states,
                encoder_attention_mask=first_decoder_encoder_hidden_states_mask,
                second_encoder_hidden_states=None,
                second_encoder_attention_mask=None,
                past_key_values=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            decoder_hidden_states_mask = first_decoder_generate_output['generate_hidden_states_mask']
            decoder_hidden_states = first_decoder_generate_output['generate_result']
            decoder_encoder_output = self.model.encode(
                input_ids=decoder_hidden_states,
                attention_mask=decoder_hidden_states_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size, -1,
                                                                                                        self.base_model.config.d_model)
            second_decoder_second_encoder_hidden_states_mask = decoder_hidden_states_mask.view(batch_size, -1)

        elif self.args.inference_type == 'ground_truth_ids':
            if len(decoder_target_ids.shape) == 3:
                batch_size, knowledge_num, seq_len = decoder_target_ids.shape
            else:
                knowledge_num = 1
                batch_size, seq_len = decoder_target_ids.shape

            decoder_target_ids = decoder_target_ids.view(batch_size * knowledge_num, seq_len)
            decoder_target_mask = decoder_target_mask.view(batch_size * knowledge_num, seq_len)

            decoder_encoder_output = self.model.encode(
                input_ids=decoder_target_ids,
                attention_mask=decoder_target_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_decoder_second_encoder_hidden_states = decoder_encoder_output.last_hidden_state.view(batch_size,
                                                                                                        knowledge_num * seq_len,
                                                                                                        -1)
            second_decoder_second_encoder_hidden_states_mask = decoder_target_mask.view(batch_size,
                                                                                               knowledge_num * seq_len)



        elif self.args.inference_type == 'no_inference':
            second_decoder_second_encoder_hidden_states = None
            second_decoder_second_encoder_hidden_states_mask = None

        if self.args.one_crossattention and second_decoder_second_encoder_hidden_states is not None:
            second_decoder_first_encoder_hidden_states = torch.cat([second_decoder_first_encoder_hidden_states, second_decoder_second_encoder_hidden_states], dim=1)
            second_decoder_first_encoder_hidden_states_mask = torch.cat([second_decoder_first_encoder_hidden_states_mask, second_decoder_second_encoder_hidden_states_mask], dim=1)
            second_decoder_second_encoder_hidden_states = None
            second_decoder_second_encoder_hidden_states_mask = None


        if self.args.num_beam == -1:

            second_decoder_output = autoregress_generation(
                self.second_decode,
                second_decoder_input_ids,
                second_decoder_input_mask,
                second_decoder_current_length,
                max_target_length,
                self.config.eos_token_id,
                encoder_hidden_states=second_decoder_first_encoder_hidden_states,
                encoder_attention_mask=second_decoder_first_encoder_hidden_states_mask,
                second_encoder_hidden_states=second_decoder_second_encoder_hidden_states,
                second_encoder_attention_mask=second_decoder_second_encoder_hidden_states_mask,
                past_key_values=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            second_generate_result = second_decoder_output['generate_result']

        else:

            second_generate_result = self.generate(
                decoder_input_ids=second_decoder_input_ids,
                decoder_attention_mask=second_decoder_input_mask,
                encoder_outputs=BaseModelOutput(last_hidden_state=second_decoder_first_encoder_hidden_states),
                attention_mask=second_decoder_first_encoder_hidden_states_mask,
                max_length=max_target_length,
                eos_token_id=self.config.eos_token_id,
                past_key_values=None,
                use_cache=use_cache,
                num_beams=self.args.num_beam,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

        return second_generate_result


    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def prepare_inputs_for_generation(self, decoder_input_ids, decoder_attention_mask=None, encoder_outputs=None, attention_mask=None, past=None, **kwargs):
        return_dict = {}
        return_dict['encoder_hidden_states'] = encoder_outputs.last_hidden_state
        return_dict['encoder_attention_mask'] = attention_mask
        if past is not None:
            return_dict['input_ids'] = decoder_input_ids[:, -1:]
        else:
            return_dict['input_ids'] = decoder_input_ids
        return_dict['input_mask'] = decoder_attention_mask
        return_dict['past_key_values'] = past
        return_dict['use_cache'] = kwargs['use_cache']
        return_dict['generate'] = True

        return return_dict