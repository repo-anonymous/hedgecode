import json
import os

import torch

if __name__ == '__main__':

    languages = ["ruby", "javascript", "php", "go", "java", "python"]
    for lang in languages:
        saved_dir = f"../detection dataset/{lang}"
        print(f"saved idr: {saved_dir}")
        dataset_arr = ["test", "train", "valid"]
        topK = 2

        for dataset in dataset_arr:
            print("**************************************************************************")
            dataset_file_name = f"../detection dataset/{lang}/{dataset}/{dataset}_top{topK}.jsonl"
            print(f"dataset file: {dataset_file_name}")
            pos_pairs = []
            neg_pairs = []
            sep_token = ' [SEP] '

            with open(dataset_file_name, 'r', encoding="utf-8") as q_f:

                for line in q_f:
                    filter_data = {}
                    json_obj = json.loads(line)
                    description = json_obj["code_pos_doc"]
                    pos_code = json_obj["code_pos"]
                    neg_code_1 = json_obj["code_neg_1"]
                    neg_code_2 = json_obj["code_neg_2"]
                    # neg_code_3 = json_obj["code_neg_3"]

                    pos_pair = description + sep_token + pos_code
                    neg_pair_1 = description + sep_token + neg_code_1
                    neg_pair_2 = description + sep_token + neg_code_2
                    # neg_pair_3 = pos_token + neg_token + description + sep_token + neg_code_3

                    pos_pairs.append(pos_pair)
                    neg_pairs.append(neg_pair_1)
                    neg_pairs.append(neg_pair_2)
                    # neg_pairs.append(neg_pair_3)

            print(f"number of pos pairs dataset: {len(pos_pairs)}")
            print(f"number of neg pairs dataset: {len(neg_pairs)}")

            pos_labels = [1]*len(pos_pairs)
            neg_labels = [0]*len(neg_pairs)

            datadict = {}

            datadict['pos_pairs'] = pos_pairs
            datadict['neg_pairs'] = neg_pairs
            datadict['pos_labels'] = pos_labels
            datadict['neg_labels'] = neg_labels

            print(f"pair dataset length: {str(len(pos_pairs) + len(neg_pairs))}")

            saved_pair_dir = f"../detection dataset/{lang}/{dataset}"
            if not os.path.exists(saved_pair_dir):
                os.makedirs(saved_pair_dir)

            torch.save(datadict, f"{saved_pair_dir}/{dataset}.h5")
            print(f"saved file: {saved_pair_dir}/{dataset}.h5")