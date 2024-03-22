import random
from random import choice
import torch
import torch.nn as nn
import torch.nn.functional as F

def replace_tokens(inputs, speical_token_ids, tokenizer, mlm_probability):
    labels = inputs.clone()
    probability_matrix = torch.full(labels.shape, 0.0).to(inputs.device)
    probability_matrix.masked_fill_(~labels.eq(speical_token_ids).to(inputs.device), value=mlm_probability)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool().to(inputs.device) & masked_indices
    inputs[indices_replaced] = speical_token_ids
    return inputs, labels

def augment_data(input_ids, tokenizer, mlm_probability=0.2):
    random_percent = random.randint(1, 98)
    _transformations_ids = input_ids.clone()
    vocab_list = tokenizer.get_vocab()
    drop_token_ids = [tokenizer.mask_token_id, tokenizer.pad_token_id, tokenizer.cls_token_id,
                        tokenizer.sep_token_id, tokenizer.unk_token_id, tokenizer.convert_tokens_to_ids('[NEG]'), tokenizer.convert_tokens_to_ids('[POS]')]
    vocab_list = [token for token in vocab_list.values() if token not in drop_token_ids]
    if 0 < random_percent < 50:
        _transformations_ids[:, 5:-2], _ = replace_tokens(input_ids.clone()[:, 5:-2], tokenizer.mask_token_id, tokenizer,
                                                        mlm_probability)
    elif 50 <= random_percent < 100:
        choice_token_id = choice(vocab_list)
        _transformations_ids[:, 5:-2], _ = replace_tokens(input_ids.clone()[:, 5:-2], choice_token_id, tokenizer,
                                                        mlm_probability)
    return _transformations_ids

class HCLModel(nn.Module):
    def __init__(self, encoder, args, tokenizer, hidden_size):
        super(HCLModel, self).__init__()
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.fc = nn.Linear(hidden_size, 2)
        self.dropout = nn.Dropout(0.5)
        self.num_classes = 2

        if self.training and encoder.training:
            self.batchsize = args.batch_size
            self.K = args.batch_size * 5
            self.register_buffer("queue", torch.randn(768, self.K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    def forward(self, input_ids):
        if self.training:
            # anchor
            outputs = self.encoder(**input_ids).last_hidden_state
            cls_feats = outputs[:, 0, :]
            label_feats = outputs[:, 1:self.num_classes + 1, :]
            predicts = self.dropout(self.fc(cls_feats))

            # data_augment
            pos_inputs_ids = self._data_augment(input_ids["input_ids"])
            pos_outputs = self.encoder(pos_inputs_ids, attention_mask=input_ids["attention_mask"]).last_hidden_state

            pos_cls_feats = pos_outputs[:, 0, :]
            pos_label_feats = pos_outputs[:, 1:self.num_classes + 1, :]
            pos_predicts = self.dropout(self.fc(pos_cls_feats))

            # Dynamic hard negative sampling
            neg_cls_feats, gamms = self._compute_similar(self.queue, cls_feats)
            neg_predicts = self.dropout(self.fc(cls_feats))

            fin_outputs = {
                'predicts': predicts,
                'cls_feats': cls_feats,
                'label_feats': label_feats,
                'pos_predicts': pos_predicts,
                'pos_cls_feats': pos_cls_feats,
                'pos_label_feats': pos_label_feats,
                'neg_predicts': neg_predicts,
                'neg_cls_feats': neg_cls_feats,
                'gamms': gamms
            }
            self._dequeue_and_enqueue(cls_feats)
            return fin_outputs
        else:
            outputs = self.encoder(**input_ids).last_hidden_state
            cls_feats = outputs[:, 0, :]
            label_feats = outputs[:, 1:self.num_classes + 1, :]
            predicts = self.dropout(self.fc(cls_feats))
            fin_outputs = {
                'predicts': predicts,
                'cls_feats': cls_feats,
                'label_feats': label_feats,
            }
            return fin_outputs

    @torch.no_grad()
    def _data_augment(self, input_ids):
        return augment_data(input_ids, self.tokenizer, mlm_probability=0.2)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, _vec):
        nl_size = _vec.shape[0]
        ptr = int(self.queue_ptr)
        if nl_size == self.batchsize:
            self.queue[:, ptr:ptr + nl_size] = _vec.T
            ptr = (ptr + nl_size) % self.K
            self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _compute_similar(self, queue, _vec):
        inner_products = torch.einsum('bd,cd->bc', _vec, queue.T)
        inner_products = F.normalize(inner_products, p=2, dim=1)
        max_similar_indices = torch.argmax(inner_products, dim=1)
        similar = queue.T[max_similar_indices]
        max_inner_products = torch.diagonal(inner_products[:, max_similar_indices])
        gamms = 1 - max_inner_products
        return similar, gamms