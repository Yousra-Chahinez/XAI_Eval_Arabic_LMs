def perturb_salient_tokens(token_ids, saliencies, threshold, tokenizer, mask=True):
    """
    Mask the most salient tokens in a sequence based on given saliency scores and a threshold.
    """
    if len(token_ids) != len(saliencies):
        raise ValueError("Length of token_ids and saliencies must match.")

    n_tokens = len([_t for _t in token_ids if _t != tokenizer.pad_token_id])
    k = int((threshold / 100) * n_tokens)

    sorted_idx = np.array(saliencies).argsort()[::-1]
    new_token_ids = token_ids[:]

    if mask and k > 0:
        num_masked = 0
        for _id in sorted_idx:
            if _id < n_tokens and token_ids[_id] != tokenizer.pad_token_id:
                new_token_ids[_id] = tokenizer.mask_token_id
                num_masked += 1
                if num_masked == k:
                    break

    return new_token_ids

def predict(model, input_ids, device, attention_mask=None, token_type_ids=None):
    """Predict the label for a given input sequence."""
    input_ids = torch.tensor([input_ids]).to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
    probas = F.softmax(outputs.logits, dim=-1)
    pred = torch.argmax(probas, dim=-1).cpu().item()
    return pred
