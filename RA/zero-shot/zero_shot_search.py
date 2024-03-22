import argparse
import json
import logging
import os
import time
from functools import partial
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from model.HCLModel import HCLModel
from tqdm import tqdm

class DetectionDataset(Dataset):
    def __init__(self, model, tokenizer, dataset_file):
        self.pair = []
        self.label = []
        self.other_info = []
        self.same_query = []

        dataset = []
        with open(dataset_file, 'r', encoding="utf-8") as q_f:
            for line in q_f:
                json_obj = json.loads(line)
                dataset.append(json_obj)

        idx = 0
        for line in dataset:
            idx += 1
            if idx % 10000 == 0:
                print(idx)
            for raw_data in line:
                pair = raw_data["pair"]
                label = raw_data["label"]

                self.pair.append(pair)
                self.label.append(label)
                self.other_info.append(raw_data)
            self.same_query.append(line)

    def __len__(self):
        return len(self.pair)

    def __getitem__(self, i):
        return self.pair[i], self.label[i]

    def get_other_info(self):
        return self.other_info

    def get_same_query(self):
        return self.same_query

def my_collate(batch, tokenizer, method, num_classes):
    tokens, label = map(list, zip(*batch))
    text_ids = tokenizer(tokens, return_tensors='pt', padding=True, truncation=True, max_length=128, is_split_into_words=True, add_special_tokens=True)
    return text_ids, torch.tensor(label)

def chunk_list(lst, chunk_size):
    """Split a list into chunks of given size."""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument("--language", default=None, type=str, required=True, help="The programming language.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--pair_dataset_file", default=None, type=str, required=True, help="Query file name.")
    parser.add_argument('--encoder', type=str, default='codebert', choices=['codebert', 'unixcoder', 'cocosoda'])
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--plugin_checkpoint_path", default=None, type=str, required=True, help="trained detector.")
    parser.add_argument("--topK", default=100, type=int, help="Recall topK codes.")

    args = parser.parse_args()

    saved_dir = args.output_dir
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

    logger.info(args)
    batch_size = args.batch_size
    lang = args.language
    output_dir = args.output_dir
    encoder_name = args.encoder
    plugin_checkpoint_path = args.plugin_checkpoint_path
    topK = args.topK
    pair_dataset_file = args.pair_dataset_file

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("device: %s", device)

    if encoder_name == 'cocosoda':
        tokenizer = RobertaTokenizer.from_pretrained("DeepSoftwareAnalytics/CoCoSoDa")
        encoder = RobertaModel.from_pretrained(f"DeepSoftwareAnalytics/CoCoSoDa")
    if encoder_name == 'unixcoder':
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/unixcoder-base")
        encoder = RobertaModel.from_pretrained(f"microsoft/unixcoder-base")
    if encoder_name == 'codebert':
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        encoder = RobertaModel.from_pretrained(f"microsoft/codebert-base")

    encoder.eval()
    encoder.to(device)

    if plugin_checkpoint_path is not None:
        logger.info(plugin_checkpoint_path)
        hidden_size = encoder.config.hidden_size
        detector = HCLModel(encoder, args, None, hidden_size=hidden_size).to(device)
        state_dict = torch.load(plugin_checkpoint_path)
        detector.load_state_dict(state_dict, strict=False)
        model = detector.encoder
    logger.info(f"plugin_checkpoint_path : {plugin_checkpoint_path}")

    detector.eval()
    detector.to(device)

    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

    pairs_num = sum(1 for line in open(pair_dataset_file, 'r', encoding="utf-8"))
    logger.info(f"length of source test dataset：{pairs_num}")

    detection_dataset = DetectionDataset(detector, tokenizer, pair_dataset_file)

    collate_fn = partial(my_collate, tokenizer=tokenizer, method=None, num_classes=2)
    data_loader = DataLoader(detection_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    total_acc = 0
    y_trues = []
    y_preds = []
    probility = []
    with torch.no_grad():
        for text, label in tqdm(data_loader):
            text = text.to(device)
            label = label.to(device)
            outputs = detector(text)
            logits = outputs['predicts']
            y_trues.append(label.tolist())
            _prob, _index = torch.max(logits, dim=-1)
            probility.append(_prob.tolist())
            y_preds.append(_index.tolist())

    y_trues = [elem for row in y_trues for elem in row]
    y_preds = [elem for row in y_preds for elem in row]
    probility = [elem for row in probility for elem in row]
    logger.info(f'y_trues :{len(y_trues)}')
    logger.info(f'y_preds :{len(y_preds)}')
    logger.info(f'probility :{len(probility)}')

    y_trues = chunk_list(y_trues, topK)
    y_preds = chunk_list(y_preds, topK)
    probility = chunk_list(probility, topK)

    MRR = []
    Acc = 0
    for truth, preds, prob in zip(y_trues, y_preds, probility):
        triplets = zip(truth, preds, prob)
        sorted_triplets = sorted(triplets, key=lambda x: x[2], reverse=True)
        count_n = 0
        for ind, (t, p, pro) in enumerate(sorted_triplets):  # 0,0,1,2
            if t == p:
                Acc += 1
            if t != 1 or p != 1:
                if p == 0:
                    count_n += 1
                continue
            else:
                rank = (ind - count_n + 1)
                MRR.append(1 / rank)
                continue

    logger.info(f"ACC: {Acc / (len(y_trues) * topK)}")
    logger.info(f"MRR: {sum(MRR) / len(y_trues)}")































