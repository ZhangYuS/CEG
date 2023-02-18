import torch

def remove_blank(sentence_list):

    def is_chinese(ch):
        if '\u4e00' <= ch <= '\u9fa5':
            return True
        else:
            return False

    ret_sentence_list = []
    for i in range(len(sentence_list)):
        sentence = sentence_list[i].strip()
        result_sentence = ''
        for ch_idx, ch in enumerate(sentence):
            if ch == ' ':
                if not is_chinese(sentence[ch_idx - 1]) and not is_chinese(sentence[ch_idx + 1]):
                    result_sentence += ch
            else:
                result_sentence += ch
        ret_sentence_list.append(result_sentence)
    return ret_sentence_list


def autoregress_generation(forward_function, input_ids, input_mask, current_length, max_target_length, eos_token_id, **kwargs):
    batch_size = input_ids.shape[0]
    current_length = current_length.view(batch_size)
    finish = torch.ones(batch_size, 1, device=input_ids[0].device)
    total_result = input_ids[:, 1:]
    total_hidden_states = None
    total_hidden_states_mask = None
    generate_hidden_states = None
    generate_hidden_states_mask = None
    generate_pro = None
    generate_result = None
    generate_length = torch.zeros_like(finish)

    past_key_values = None
    kwargs['past_key_values'] = past_key_values
    for step in range(max_target_length):
        decoder_output, hidden_state, lm_logits, lm_pro = forward_function(
            input_ids,
            input_mask,
            **kwargs
        )
        kwargs['past_key_values'] = decoder_output.past_key_values
        kwargs['position_ids'] = current_length.unsqueeze(1)
        if step == 0:

            next_token_logits = hidden_state[torch.arange(batch_size), (current_length - 1), :]
            next_token_score = lm_logits[torch.arange(batch_size), (current_length - 1), :]
            next_token_pro = lm_pro[torch.arange(batch_size), (current_length - 1), :]
        else:
            next_token_logits = hidden_state[:, -1, :]
            next_token_score = lm_logits[:, -1, :]
            next_token_pro = lm_pro[:, -1, :]
        input_ids = torch.argmax(next_token_score, dim=-1, keepdim=True)

        total_result = torch.cat([total_result, input_ids], dim=1)
        total_result[torch.arange(batch_size), current_length - 1] = input_ids.view(-1)

        if total_hidden_states_mask is None:
            total_hidden_states_mask = input_mask
        else:
            total_hidden_states_mask = torch.cat([total_hidden_states_mask, torch.zeros(batch_size, 1, device=input_ids.device)], dim=1)
            total_hidden_states_mask[torch.arange(batch_size), current_length - 1] = finish.view(-1)

        if total_hidden_states is None:
            total_hidden_states = hidden_state
        else:
            total_hidden_states = torch.cat([total_hidden_states, next_token_logits.unsqueeze(1)], dim=1)
            total_hidden_states[torch.arange(batch_size), current_length - 1] = next_token_logits

        if generate_result is None:
            generate_result = input_ids
        else:
            generate_result = torch.cat([generate_result, input_ids], dim=1)

        if generate_hidden_states is None:
            generate_hidden_states = next_token_logits.unsqueeze(1)
        else:
            generate_hidden_states = torch.cat(
                [generate_hidden_states, next_token_logits.unsqueeze(1)], dim=1)

        if generate_hidden_states_mask is None:
            generate_hidden_states_mask = finish
        else:
            generate_hidden_states_mask = torch.cat([generate_hidden_states_mask, finish], dim=1)

        if generate_pro is None:
            generate_pro = next_token_pro.unsqueeze(1)
        else:
            generate_pro = torch.cat([generate_pro, next_token_pro.unsqueeze(1)], dim=1)

        finish = finish * (input_ids != eos_token_id).long()
        generate_length += finish
        current_length += 1
        input_mask = torch.cat([input_mask, finish], dim=1)

        if torch.sum(finish) == 0:
            break

    return {
        'total_result': total_result,
        'total_hidden_states': total_hidden_states,
        'total_hidden_states_mask': total_hidden_states_mask,
        'generate_hidden_states': generate_hidden_states,
        'generate_hidden_states_mask': generate_hidden_states_mask,
        'generate_pro': generate_pro,
        'generate_result': generate_result,
        'generate_length': generate_length
    }