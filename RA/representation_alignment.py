from model.HCLModel import HCLModel
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
import os
from torch.utils.data import Dataset
from functools import partial
from sklearn.metrics import accuracy_score
import copy
import logging
import time
import argparse

class ContrastiveDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        text = []
        label = []
        self.sep_token = ['[sep]']
        self.label_list = ['[pos]', '[neg]']

        for i, data in enumerate(texts):
            tokens = data.lower().split(' ')
            label_id = labels[i]
            text.append(self.label_list + self.sep_token + tokens)
            label.append(label_id)
        self.text = text
        self.label = label

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = self.text[index]
        label = self.label[index]
        return text, label

def my_collate(batch, tokenizer, method, num_classes, args):
    tokens, label = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                          padding=True,
                          truncation=True,
                          max_length= (args.nl_length + args.code_length - 2),
                          is_split_into_words=True,
                          add_special_tokens=True,
                          return_tensors='pt')
    return text_ids, torch.tensor(label)

class hedgeLoss(nn.Module):
    def __init__(self, alpha, temp, loss_type):
        super().__init__()
        self.xent_loss = nn.CrossEntropyLoss()
        self.alpha = alpha
        self.temp = temp
        self.margin = 1.0

    def forward(self, outputs, targets):
        if self.training and loss_type == 'hcl':
            anchor_cls_feats = self.normalize_feats(outputs['cls_feats'])
            anchor_label_feats = self.normalize_feats(outputs['label_feats'])
            neg_cls_feats = self.normalize_feats(outputs['neg_cls_feats'])
            pos_cls_feats = self.normalize_feats(outputs['pos_cls_feats'])
            pos_label_feats = self.normalize_feats(outputs['pos_label_feats'])

            normed_pos_label_feats = torch.gather(pos_label_feats, dim=1, index=targets['label'].reshape(-1, 1, 1).expand(-1, 1, pos_label_feats.size(-1))).squeeze(1)
            normed_anchor_label_feats = torch.gather(anchor_label_feats, dim=1, index=targets['label'].reshape(-1, 1, 1).expand(-1, 1, anchor_label_feats.size(-1))).squeeze(1)
            normed_neg_label_feats = torch.mul(normed_anchor_label_feats, outputs['gamms'].unsqueeze(1))

            ce_loss = (1 - self.alpha) * self.xent_loss(outputs['predicts'], targets['label'])
            cl_loss_1 = 0.5 * self.alpha * self.hedge_loss(anchor_cls_feats, normed_pos_label_feats, normed_neg_label_feats)  # data view
            cl_loss_2 = 0.5 * self.alpha * self.hedge_loss(normed_anchor_label_feats, pos_cls_feats, neg_cls_feats)  # classifier view
            return ce_loss + cl_loss_1 + cl_loss_2
        else:
            ce_loss = self.xent_loss(outputs['predicts'], targets['label'])
            return ce_loss

    def hedge_loss(self, anchor, positive, negative):
        sim_pos = torch.sum(anchor * positive, dim=-1) / torch.norm(anchor) / torch.norm(positive)
        sim_neg = torch.sum(anchor * negative, dim=-1) / torch.norm(anchor) / torch.norm(negative)
        loss = -torch.mean(torch.log(torch.exp(sim_pos / self.temp) / torch.sum(torch.exp(sim_neg / self.temp), dim=-1)))
        return loss

    def normalize_feats(self, _feats):
        return F.normalize(_feats, dim=-1)

def train(args, model, train_date, logger, train_loader, optimizer, criterion, device, valid_loader):
    best_valid_loss = np.inf
    best_valid_accuracy = 0.0

    for epoch in range(args.num_train_epochs):
        model.train()
        criterion.train()
        total_loss = 0
        total_acc = 0
        bar = tqdm(enumerate(train_loader))
        for batch_idx, (text, label) in bar:
            text = text.to(device)
            label = label.to(device)
            targets = {
                'label': label
            }
            outputs = model(text)
            loss = criterion(outputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logits1 = outputs['predicts']
            total_loss += loss.item()
            total_acc += accuracy_score(torch.argmax(logits1, dim=1).tolist(), label.tolist())

            if batch_idx % 500 == 0:
                logger.info(
                    f'Epoch {epoch} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Train Accuracy: {accuracy_score(torch.argmax(logits1, dim=1).tolist(), label.tolist()):.4f}')
                print(
                    f'Epoch {epoch} - Batch {batch_idx}/{len(train_loader)} - Loss: {loss.item():.4f} - Train Accuracy: {accuracy_score(torch.argmax(logits1, dim=1).tolist(), label.tolist()):.4f}')

        train_loss = total_loss / len(train_loader)
        train_acc = total_acc / len(train_loader)
        logger.info(f'Epoch {epoch} - Train loss: {train_loss:.4f} - Train accuracy: {train_acc:.4f}')

        if valid_loader is not None:
            valid_loss, valid_acc = validate(model, criterion, device, valid_loader)
            logger.info(f'Epoch {epoch} - Valid loss: {valid_loss:.4f} - Valid accuracy: {valid_acc:.4f}')

        # early stop
        patience = 10
        counter = 0
        if valid_loss <= best_valid_loss:
            logger.info(f'best valid loss has improved ({best_valid_loss}---->{valid_loss})')
            best_valid_loss = valid_loss
            best_valid_accuracy = valid_acc
            best_model_state = copy.deepcopy(model.state_dict())
            torch.save(best_model_state, f'./{args.output_dir}/{args.language}/{args.encoder}/{args.loss_type}/detector.pth')
            logger.info('A new best model state  has saved')
        else:
            counter += 1
        if counter >= patience:
            logger.info("Early stopping. No improvement in {} epochs.".format(patience))
            break

    logger.info('Training Finish !!!!!!!!')
    logger.info(f'best valid loss == {best_valid_loss}, best valid accuracy == {best_valid_accuracy}')

    return model

def validate(model, criterion, device, valid_loader):
    print("________________valid_______________")
    model.eval()
    criterion.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for text, label in valid_loader:
            text = text.to(device)
            label = label.to(device)
            targets = {
                'label': label
            }
            outputs = model(text)
            loss = criterion(outputs, targets)
            logits = outputs['predicts']
            total_loss += loss.item()
            total_acc += accuracy_score(torch.argmax(logits, dim=1).tolist(), label.tolist())
    valid_loss = total_loss / len(valid_loader)
    valid_acc = total_acc / len(valid_loader)
    return valid_loss, valid_acc


def read_datasets(lang, logger, args):
    dataset_arr = ["test", "train", "valid"]

    train_texts = []
    test_texts = []
    valid_texts = []
    train_labels = []
    test_labels = []
    valid_labels = []

    for dataset in dataset_arr:
        dataset_file_name = f"{args.detection_dir}/{lang}/{dataset}/{dataset}.h5"
        dt = torch.load(dataset_file_name)

        posinput = dt['pos_pairs']
        neginput = dt['neg_pairs']
        poslabels = dt['pos_labels']
        neglabels = dt['neg_labels']

        texts = posinput + neginput
        labels = poslabels + neglabels
        _texts = np.array(texts)
        _labels = np.array(labels)

        logger.info(f"{dataset} dataset file path: {dataset_file_name} - length of {dataset} dataset: {len(texts)}")

        perm = np.random.permutation(len(_texts))
        texts_shuffled = _texts[perm]
        labels_shuffled = _labels[perm]

        if dataset == "train":
            train_texts = texts_shuffled
            train_labels = labels_shuffled
        if dataset == "test":
            test_texts = texts_shuffled
            test_labels = labels_shuffled
        if dataset == "valid":
            valid_texts = texts_shuffled
            valid_labels = labels_shuffled

    return train_texts, test_texts, valid_texts, train_labels, test_labels, valid_labels

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", default=None, type=str, required=True, help="The programming language.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--detection_dir", default=None, type=str, required=True,
                        help="The folder of detection pair datasets.")
    parser.add_argument('--encoder', type=str, default='codebert', choices=['codebert', 'unixcoder', 'cocosoda'])
    parser.add_argument("--nl_length", default=128, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=256, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument('--loss_type', type=str, default='ce', choices=['ce', 'hcl'], help="Loss function type.")
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--learning_rate", default=1e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=1, type=int, help="Total number of training epochs to perform.")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lr = args.learning_rate
    epochs = args.num_train_epochs
    batch_size = args.batch_size
    loss_type = args.loss_type
    lang = args.language
    output_dir = args.output_dir
    encoder_name = args.encoder
    alpha = 0.5
    temp = 0.1

    saved_dir = f"{output_dir}/{lang}/{encoder_name}/{loss_type}"
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    if encoder_name == 'cocosoda':
        tokenizer = RobertaTokenizer.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
        encoder = RobertaModel.from_pretrained(f"DeepSoftwareAnalytics/CoCoSoDa")
    if encoder_name == 'unixcoder':
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
        encoder = RobertaModel.from_pretrained(f"microsoft/unixcoder-base")
    if encoder_name == 'codebert':
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        encoder = RobertaModel.from_pretrained(f"microsoft/codebert-base")

    special_tokens = {
        "additional_special_tokens": ['[POS]', '[NEG]']
    }

    tokenizer.add_special_tokens(special_tokens)

    current_time = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    logger = logging.getLogger('train_logger')
    logger.setLevel(logging.INFO)

    log_file_name = os.path.join(saved_dir, f"{current_time}.log")

    with open(log_file_name, 'w') as file:
        pass
    file_handler = logging.FileHandler(log_file_name)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


    train_texts, test_texts, val_texts, train_labels, test_labels, val_labels = read_datasets(lang, logger, args)
    train_dataset = ContrastiveDataset(train_texts, train_labels, tokenizer)
    val_dataset = ContrastiveDataset(val_texts, val_labels, tokenizer)
    test_dataset = ContrastiveDataset(test_texts, test_labels, tokenizer)

    hyper_parameter = f"hyper-parameter: - lr: {lr}; epochs: {epochs}; batch_size: {batch_size}"
    logger.info(hyper_parameter)
    logger.info(f"encoder_name: {encoder_name}")
    hidden_size = encoder.config.hidden_size

    collate_fn = partial(my_collate, tokenizer=tokenizer, method=None, num_classes=2, args=args)
    train_loader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    encoder.train()
    model = HCLModel(encoder, args=args, tokenizer=tokenizer, hidden_size=hidden_size).to(device)

    logger.info(f"model structure: ")
    logger.info(f"=======================================================================================")
    logger.info(model)
    logger.info(f"=======================================================================================")

    # training
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.05, betas=(0.9, 0.99), eps=1e-8, amsgrad=True)
    criterion = hedgeLoss(alpha, temp, loss_type)
    train_date = ''.join(str(datetime.now().date()).split("-"))
    model = train(args, model, train_date, logger, train_loader, optimizer, criterion, device, valid_loader=val_loader)

    # evaluation
    encoder.eval()
    model.eval()
    criterion.eval()
    total_acc = 0
    y_true = []
    y_pred = []
    with torch.no_grad():
        for text, label in test_loader:
            text = text.to(device)
            label = label.to(device)

            outputs = model(text)
            logits = outputs['predicts']
            y_true.append(label)
            y_pred.append(torch.argmax(logits, -1))
            total_acc += accuracy_score(torch.argmax(logits, dim=1).tolist(), label.tolist())

    test_acc = total_acc / len(test_loader)

    logger.info(f'Test accuracy: {test_acc:.4f}')