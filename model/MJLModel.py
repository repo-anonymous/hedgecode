import torch.nn as nn
import torch
from torch.nn import CrossEntropyLoss
import random
import torch.nn.functional as F

def hedge_contrastive_loss(outputs, targets, temperature=0.1, alpha =0.5):
    xent_loss = nn.CrossEntropyLoss()

    anchor_cls_feats = normalize_feats(outputs['cls_feats'])
    anchor_label_feats = normalize_feats(outputs['label_feats'])
    neg_cls_feats = normalize_feats(outputs['neg_cls_feats'])
    pos_cls_feats = normalize_feats(outputs['pos_cls_feats'])
    pos_label_feats = normalize_feats(outputs['pos_label_feats'])
    normed_pos_label_feats = torch.gather(pos_label_feats, dim=1, index=targets['label'].reshape(-1, 1, 1).expand(-1, 1, pos_label_feats.size(-1))).squeeze(1)
    normed_anchor_label_feats = torch.gather(anchor_label_feats, dim=1,index=targets['label'].reshape(-1, 1, 1).expand(-1, 1, anchor_label_feats.size(-1))).squeeze(1)
    normed_neg_label_feats = torch.mul(normed_anchor_label_feats, outputs['gamms'].unsqueeze(1))

    ce_loss = (1 - alpha) * (xent_loss(outputs['predicts'], targets['label']))
    cl_loss_1 = 0.5 * alpha * hedge_loss(anchor_cls_feats, normed_pos_label_feats, normed_neg_label_feats, temperature)  # data view
    cl_loss_2 = 0.5 * alpha * hedge_loss(normed_anchor_label_feats, pos_cls_feats, neg_cls_feats, temperature)  # classifier view
    return ce_loss + cl_loss_1 + cl_loss_2

def hedge_loss(anchor, positive, negative, temperature):
    sim_pos = torch.sum(anchor * positive, dim=-1) / torch.norm(anchor) / torch.norm(positive)
    sim_neg = torch.sum(anchor * negative, dim=-1) / torch.norm(anchor) / torch.norm(negative)
    loss = -torch.mean(torch.log(torch.exp(sim_pos / temperature) / torch.sum(torch.exp(sim_neg / temperature), dim=-1)))
    return loss

def normalize_feats( _feats):
    return F.normalize(_feats, dim=-1)

def self_supervised_contrastive_loss(input1, input2, temperature=0.1):
    batch_size = input1.size(0)
    cos_sim = F.cosine_similarity(input1.unsqueeze(1), input2.unsqueeze(0), dim=2)
    positives = torch.diag(cos_sim).view(batch_size, 1)
    mask = torch.eye(batch_size, dtype=torch.bool, device=input1.device)
    negatives = cos_sim.masked_select(~mask).view(batch_size, -1)
    logits = torch.cat([positives, negatives], dim=1) / temperature
    labels = torch.zeros(batch_size, dtype=torch.long, device=input1.device)
    loss = F.cross_entropy(logits, labels)
    return loss

def connection_and_padding(text, code, tokenizer):
    dim_text = next((i for i, dim in enumerate(text) if dim == 1), -1)
    dim_code = next((i for i, dim in enumerate(code) if dim == 1), -1)
    prefix_tensor = torch.tensor([tokenizer.cls_token_id, tokenizer.convert_tokens_to_ids("[POS]"), tokenizer.convert_tokens_to_ids("[NEG]")]).to(code.device)
    sep_tensor = torch.tensor(tokenizer.sep_token_id).unsqueeze(0).to(code.device)
    concat_tensor_pre = torch.cat((prefix_tensor, text[1:dim_text]), dim=0).to(code.device)
    concat_tensor_last = torch.cat((sep_tensor, code[1:dim_code]), dim=0).to(code.device)
    concat_tensor = torch.cat((concat_tensor_pre, concat_tensor_last), dim=0).to(code.device)
    padding_size = text.size()[0] + code.size()[0] - concat_tensor.size(0)
    padding_tensor = torch.ones(padding_size, dtype=torch.int).to(code.device)
    padded_tensor = torch.cat((concat_tensor, padding_tensor), dim=0)
    return padded_tensor

def build_pairs(bs, code_inputs, nl_inputs, tokenizer):
    shape = (bs, int(code_inputs.shape[1] + nl_inputs.shape[1]))
    pos_pairs = torch.empty(shape, dtype=torch.long)
    # neg_pairs = torch.empty(shape, dtype=torch.long)
    for index in range(bs):
        # negative_index = random.choice([i for i in range(bs) if i != index])
        # neg_code = code_inputs[negative_index]
        pos_code = code_inputs[index]
        text = nl_inputs[index]
        pos_pairs[index] = connection_and_padding(text, pos_code, tokenizer)
        # neg_pairs[index] = connection_and_padding(text, neg_code, tokenizer)
    # return pos_pairs.to(code_inputs.device), neg_pairs.to(code_inputs.device)
    return pos_pairs.to(code_inputs.device)

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
        choice_token_id = random.choice(vocab_list)
        _transformations_ids[:, 5:-2], _ = replace_tokens(input_ids.clone()[:, 5:-2], choice_token_id, tokenizer,
                                                        mlm_probability)
    return _transformations_ids

class MJLModel(nn.Module):
    def __init__(self, encoder, tokenizer, args):
        super(MJLModel, self).__init__()
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.args = args
        self.fc = nn.Linear(encoder.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.5)
        self.num_classes = 2

        if self.training:
            self.batchsize = args.train_batch_size
            self.K = args.train_batch_size * 5
            self.register_buffer("queue", torch.randn(768, self.K))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _data_augment(self, input_ids):
        return augment_data(input_ids, self.tokenizer, mlm_probability=0.2)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, _vec):
        nl_size = _vec.shape[0]  # [B*768] ==> shape[0] = B
        ptr = int(self.queue_ptr)
        if nl_size == self.batchsize:  # B
            self.queue[:, ptr:ptr + nl_size] = _vec.T  # [768*B]
            ptr = (ptr + nl_size) % self.K  # [B]
            self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _compute_similar(self, queue, _vec):
        inner_products = torch.einsum('bd,cd->bc', _vec, queue.T)  # B * 128
        inner_products = F.normalize(inner_products, p=2, dim=1)
        max_similar_indices = torch.argmax(inner_products, dim=1)
        similar = queue.T[max_similar_indices]  # [B*768]
        max_inner_products = torch.diagonal(inner_products[:, max_similar_indices])  # [B]
        gamms = 1 - max_inner_products
        return similar, gamms

    def forward(self, code_inputs=None, nl_inputs=None):
        # train stage
        if (code_inputs is not None) and (nl_inputs is not None):
            #### cs: text-code search ####
            nl_vec = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            nl_vec = (nl_vec * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(1).sum(-1)[:, None]
            torch.nn.functional.normalize(nl_vec, p=2, dim=1)

            code_vec = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            code_vec = (code_vec * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(1).sum(-1)[:, None]
            code_vec = torch.nn.functional.normalize(code_vec, p=2, dim=1)

            scores = torch.einsum("ab,cb->ac", nl_vec, code_vec)
            loss_fct = CrossEntropyLoss()
            labels = torch.arange(code_inputs.size(0), device=scores.device)
            loss_cs = loss_fct(scores, labels)

            #### ctrd: text-code relevance detection ####
            # build code-text pairs
            bs = code_inputs.size(0)
            # anchor_pairs_inputs, neg_pairs_inputs = build_pairs(bs, code_inputs, nl_inputs, self.tokenizer)
            anchor_pairs_inputs = build_pairs(bs, code_inputs, nl_inputs, self.tokenizer)
            outputs = self.encoder(anchor_pairs_inputs, attention_mask=anchor_pairs_inputs.ne(1)).last_hidden_state  # [B*length*768]
            cls_feats = outputs[:, 0, :]  # [B*768]
            label_feats = outputs[:, 1:self.num_classes + 1, :]  # [B*2*768]
            predicts = self.dropout(self.fc(cls_feats))  # [B*2]
            # data_augment
            pos_inputs_ids = self._data_augment(anchor_pairs_inputs)  # B * len * 768
            pos_outputs = self.encoder(pos_inputs_ids, attention_mask=anchor_pairs_inputs.ne(1)).last_hidden_state  # [B*len*768]
            pos_cls_feats = pos_outputs[:, 0, :]  # B*768
            pos_label_feats = pos_outputs[:, 1:self.num_classes + 1, :]  # [B*2*768]
            pos_predicts = self.dropout(self.fc(pos_cls_feats))

            # Dynamic hard negative sampling
            neg_cls_feats, gamms = self._compute_similar(self.queue, cls_feats)  # [B*768], [B]
            neg_predicts = self.dropout(self.fc(cls_feats))  # [B*2]
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
            targets = {
                'label': torch.torch.ones(bs).to(torch.int64).to(self.args.device)
            }
            self._dequeue_and_enqueue(cls_feats)

            loss_ctrd = hedge_contrastive_loss(fin_outputs, targets)
            #### ctc: code-text contrastive learning  ####
            # data augment
            augmented_code_inputs = augment_data(code_inputs, self.tokenizer).to(self.args.device)
            augmented_nl_inputs = augment_data(nl_inputs,  self.tokenizer).to(self.args.device)
            # get feature encodings
            aug_code_vec = self.encoder(augmented_code_inputs, attention_mask=augmented_code_inputs.ne(1))[1]
            aug_nl_vec = self.encoder(augmented_nl_inputs, attention_mask=augmented_nl_inputs.ne(1))[1]
            # compute self-supervised contrastive loss
            contrastive_loss_code = self_supervised_contrastive_loss(code_vec, aug_code_vec)
            contrastive_loss_nl = self_supervised_contrastive_loss(nl_vec, aug_nl_vec)
            contrastive_loss_nl_code = self_supervised_contrastive_loss(nl_vec, code_vec)
            loss_ctc = contrastive_loss_code + contrastive_loss_nl + contrastive_loss_nl_code
            loss = loss_cs + 0.1 * loss_ctc + 0.1 * loss_ctrd
            return loss

        if code_inputs is not None:
            outputs = self.encoder(code_inputs, attention_mask=code_inputs.ne(1))[0]
            outputs = (outputs * code_inputs.ne(1)[:, :, None]).sum(1) / code_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)
        else:
            outputs = self.encoder(nl_inputs, attention_mask=nl_inputs.ne(1))[0]
            outputs = (outputs * nl_inputs.ne(1)[:, :, None]).sum(1) / nl_inputs.ne(1).sum(-1)[:, None]
            return torch.nn.functional.normalize(outputs, p=2, dim=1)