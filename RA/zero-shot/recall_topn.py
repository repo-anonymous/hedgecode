import argparse
import json
import logging
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
import time
from utils.ball_tree import ball_tree
from model.HCLModel import HCLModel

class QueryDataset(Dataset):
    def __init__(self, model, tokenizer, query_file_path=None):
        self.queries = []
        self.urls = []
        self.raw_data = []
        self.datas = []

        with open(query_file_path, "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                idx += 1
                if idx % 500 == 0:
                    print(idx)
                line = line.strip()
                query_info_json = json.loads(line)

                query = query_info_json['docstring_tokens']

                url = query_info_json['url']
                self.queries.append(query)
                self.urls.append(url)
                self.raw_data.append(query_info_json)
                token = tokenizer(query, return_tensors='pt', padding=True, truncation=True, max_length=128, add_special_tokens=True)['input_ids']
                token = token.to(device)
                with torch.no_grad():
                    token_vec = model(token)[1]
                    self.datas.append(token_vec[0])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        return self.datas[i]

    def get_queries(self):
        return self.queries

    def get_urls(self):
        return self.urls

    def get_raw_data(self):
        return self.raw_data

class CodeDataset(Dataset):
    def __init__(self, model, tokenizer, code_file_path=None):
        self.datas = []
        self.codes = []
        self.urls = []
        self.other_info = []

        with open(code_file_path, "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                idx += 1
                if idx % 500 == 0:
                    print(idx)
                line = line.strip()
                code_info_json = json.loads(line)

                code = code_info_json['code_tokens']
                url = code_info_json['url']

                self.codes.append(code)
                self.urls.append(url)

                self.other_info.append(code_info_json)
                token = tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=256, add_special_tokens=True)['input_ids']
                token = token.to(device)
                with torch.no_grad():
                    token_vec = model(token)[1]
                    self.datas.append(token_vec[0])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, i):
        return self.datas[i]

    def get_codes(self):
        return self.codes

    def get_urls(self):
        return self.urls

    def get_other_info(self):
        return self.other_info

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None, type=str, required=True, help="The programming language.")
    parser.add_argument("--output_dir", default=None, type=str, required=True, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--query_file", default=None, type=str, required=True, help="Query file name.")
    parser.add_argument("--codebase_file", default=None, type=str, required=True, help="codebase file name.")
    parser.add_argument("--plugin_checkpoint_path", default=None, type=str, required=True, help="trained detector.")
    parser.add_argument('--encoder', type=str, default='codebert', choices=['codebert', 'unixcoder', 'cocosoda'])
    parser.add_argument("--batch_size", default=4, type=int, help="Batch size.")
    parser.add_argument("--topK", default=100, type=int, help="Recall topK codes.")
    args = parser.parse_args()

    saved_dir = args.output_dir
    if not os.path.exists(saved_dir):
        os.makedirs(saved_dir)

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
    query_file_name = args.query_file
    codebase_file_name = args.codebase_file

    recall_tpye = "ball_tree"

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

    query_embedding_file = f"{saved_dir}/query.pth"
    if os.path.exists(query_embedding_file):
        query_dataset = torch.load(query_embedding_file)
    else:
        query_dataset = QueryDataset(model, tokenizer, query_file_name)
        torch.save(query_dataset, query_embedding_file)

    codebase_embedding_file = f"{saved_dir}/codebase.pth"
    if os.path.exists(codebase_embedding_file):
        code_dataset = torch.load(codebase_embedding_file)
    else:
        code_dataset = CodeDataset(model, tokenizer, codebase_file_name)
        torch.save(code_dataset, codebase_embedding_file)

    query_dataloader = DataLoader(query_dataset, batch_size=batch_size)
    code_dataloader = DataLoader(code_dataset, batch_size=batch_size)

    code_vecs = []
    query_vecs = []

    bar1 = tqdm(code_dataloader)
    bar2 = tqdm(query_dataloader)
    for batch in bar1:
        code_vecs.append(batch.cpu().numpy())
    for batch in bar2:
        query_vecs.append(batch.cpu().numpy())

    code_vecs = np.concatenate(code_vecs, 0)
    query_vecs = np.concatenate(query_vecs, 0)

    logger.info(f"length of code_vecs : {len(code_vecs)}")
    logger.info(f"length of query_vecs : {len(query_vecs)}")

    logger.info("computer similar score with ball tree")

    start_time = time.time()

    if recall_tpye == "matrix":
        logger.info("computer similar score with matrix dot")
        scores = np.matmul(query_vecs, code_vecs.T)
        sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    if recall_tpye == "ball_tree":
        logger.info("computer similar score with ball tree")
        scores, sort_ids = ball_tree(code_vecs, topK, query_vecs)
        start_time = time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    logger.info("execution time：{:.2f} min".format(elapsed_minutes))
    logger.info(f"scores size: {scores.shape}")
    logger.info(f"sort_ids size: {sort_ids.shape}")

    # 10. save result
    logger.info("_________________________________________________________________")
    logger.info("save result")
    all_result_arr = []
    topK_results = np.array([row[:topK] for row in sort_ids])
    for i in range(0, len(topK_results)):
        raw_result_arr = []
        result_obj = {}
        query = query_dataset.get_queries()[i]
        url = query_dataset.get_urls()[i]
        code = query_dataset.get_raw_data()[i]["code_tokens"]

        result_obj["url"] = url
        result_obj["query"] = query
        result_obj["pair"] = ["POS", "NEG"] + query + ["SEP"] + code
        result_obj["label"] = 1
        raw_result_arr.append(result_obj)
        for j in range(0, topK):
            result_obj = {}
            result_obj["query"] = query
            search_result = code_dataset.other_info[sort_ids[i][j]]
            code_tokens = search_result["code_tokens"]
            search_url = search_result["url"]
            if url == search_url:
                continue
            else:
                result_obj["url"] = search_url
                result_obj["pair"] = ["POS", "NEG"] + query + ["SEP"] + code_tokens
                result_obj["label"] = 0
                raw_result_arr.append(result_obj)

        all_result_arr.append(raw_result_arr[:topK])

    logger.info(f"length of results : {np.array(all_result_arr).shape}")
    detection_pair_dataset_file = f"{saved_dir}/detection_pair_dataset.jsonl"
    with open(detection_pair_dataset_file, 'w', encoding="utf-8") as f_result:
        for d in all_result_arr:
            json.dump(d, f_result)
            f_result.write('\n')
        logger.info(f"saved top{topK} pair result: {detection_pair_dataset_file}")

    logger.info("finish")
    logger.info("_________________________________________________________________")