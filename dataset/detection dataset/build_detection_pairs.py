import json
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModel, BertTokenizer, BertModel
from tqdm import tqdm
import time
from ball_tree import ball_tree

class CodeDataset(Dataset):
    def __init__(self, model, tokenizer, code_file_path=None, code_idx_max=100000):
        self.datas = []
        self.codes = []
        self.other_info = []

        with open(code_file_path, "r", encoding="utf-8") as f:
            idx = 0
            for line in f:
                idx += 1
                if idx % 1000 == 0:
                    print(idx)
                if idx > code_idx_max:
                    break
                line = line.strip()
                code_info_json = json.loads(line)

                code = code_info_json['code_tokens']
                code = " ".join(code)

                self.codes.append(code)
                self.other_info.append(code_info_json)
                token = tokenizer(code, return_tensors='pt', padding=True, truncation=True, max_length=510, is_split_into_words=True, add_special_tokens=True,)['input_ids']
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

    def get_other_info(self):
        return self.other_info

if __name__ == '__main__':

    # model_name = "codebert-base"
    model_name = "bert"

    languages = ["ruby", "javascript", "php", "go", "java", "python"]
    datasets = ["test", "valid", "train"]

    if model_name == 'codebert-base':
        tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        model = AutoModel.from_pretrained("microsoft/codebert-base")
        print(model_name)
    if model_name == 'bert':
        tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        model = BertModel.from_pretrained("bert-base-cased")
        print(model_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device.type}")
    model.to(device)

    topK = 4
    batch_size = 1024
    for lang in languages:
        for dataset in datasets:
            saved_dir = f"../detection dataset/{lang}/{dataset}"
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)
            code_file_name = f"../detection dataset/pairs/{lang}/{dataset}.jsonl"

            idx_max = sum(1 for line in open(code_file_name, 'r', encoding="utf-8"))
            print(f"length of source {dataset} dataset：{idx_max}")

            pos_neg_result_file_name = f"{saved_dir}/top{topK}_similar.jsonl"

            codebase_embedding_file = f"{saved_dir}/{dataset}_{idx_max}.pth"
            if os.path.exists(codebase_embedding_file):
                print(f"already saved dataset embedding: {codebase_embedding_file}")
                code_dataset = torch.load(codebase_embedding_file)
            else:
                code_dataset = CodeDataset(model, tokenizer, code_file_name, idx_max)
                torch.save(code_dataset, codebase_embedding_file)
                print(f"saved dataset embedding: {codebase_embedding_file}")

            code_dataloader = DataLoader(code_dataset, batch_size=batch_size)

            code_vecs = []

            bar = tqdm(code_dataloader)
            bar.set_description(desc="concatenating")
            for batch in bar:
                code_vecs.append(batch.cpu().numpy())

            code_vecs = np.concatenate(code_vecs, 0)
            print(f"length of code_vecs : {len(code_vecs)}")

            print("computer similar score with ball tree")
            start_time = time.time()

            scores, sort_ids = ball_tree(code_vecs, topK, code_vecs)

            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_minutes = elapsed_time / 60
            print("execution time：{:.2f} min".format(elapsed_minutes))

            print(f"scores size: {scores.shape}")
            print(f"sort_ids size: {sort_ids.shape}")

            # 10. save result
            print("_________________________________________________________________")
            print("save result")
            f_result_arr = []
            for i in range(len(sort_ids)):
                code_pos = code_dataset.codes[i]
                code_pos_doc = " ".join(code_dataset.other_info[i]['docstring_tokens'])
                code_pos_func_name = code_dataset.other_info[i]['func_name']
                result_obj = {}
                result_obj["code_pos"] = code_pos
                result_obj["code_pos_doc"] = code_pos_doc
                result_obj["code_pos_func_name"] = code_pos_func_name

                for j in range(0, topK):
                    result_info = code_dataset.other_info[sort_ids[i][j]]
                    result_code = " ".join(result_info['code_tokens'])
                    result_name = result_info['func_name']
                    result_doc = " ".join(result_info['docstring_tokens'])
                    result_score = scores[i][j]
                    result_obj[f"NO.{j + 1}"] = str(result_code)
                    result_obj[f"score.{j + 1}"] = result_score
                    result_obj[f"func_name.{j + 1}"] = result_name
                    result_obj[f"doc.{j + 1}"] = result_doc

                f_result_arr.append(result_obj)
            print(f"length of results : {len(f_result_arr)}")

            with open(pos_neg_result_file_name, 'w', encoding="utf-8") as f_result:
                for d in f_result_arr:
                    json.dump(d, f_result)
                    f_result.write('\n')
                print(f"saved top{topK} result: {pos_neg_result_file_name}")

            print("finish")
            print("_________________________________________________________________")