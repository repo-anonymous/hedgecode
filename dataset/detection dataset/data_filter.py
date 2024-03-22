import json
import os

def joint_train(files):
    print(f"number of files: {len(files)}")
    train_dataset = []
    for file in files:
        with open(file, 'r', encoding="utf-8") as f:
            for line in f:
                json_obj = json.loads(line)
                train_dataset.append(json_obj)
    return train_dataset

if __name__ == '__main__':

    dataset_arr = ["test", "train", "valid"]
    topN = "4"
    topK = "2"
    languages = ["ruby", "javascript", "php", "go", "java", "python"]

    for lang in languages:
        for dataset in dataset_arr:
            print("**************************************************************************")
            print(f"dataset: {dataset}")
            pos_neg_file = f"../detection dataset/{lang}/{dataset}/top{topN}_similar.jsonl"
            _dataset = []
            with open(pos_neg_file, 'r', encoding="utf-8") as q_f:
                for line in q_f:
                    json_obj = json.loads(line)
                    _dataset.append(json_obj)
            print(f"source dataset length: {len(_dataset)}")

            _results = []
            for data in _dataset:
                if data["score.3"] <= 20.0:
                    filter_data = {}
                    filter_data["code_pos"] = data["code_pos"]
                    filter_data["code_pos_doc"] = data["code_pos_doc"]
                    filter_data["code_neg_1"] = data["NO.2"]
                    filter_data["code_neg_doc_1"] = data["doc.2"]
                    filter_data["code_neg_2"] = data["NO.3"]
                    filter_data["code_neg_doc_2"] = data["doc.3"]
                    _results.append(filter_data)

            print(f"filter dataset length: {len(_results)}")

            saved_dir = f"../detection dataset/{lang}/{dataset}"
            if not os.path.exists(saved_dir):
                os.makedirs(saved_dir)
            print(f"saved idr: {saved_dir}")

            dataset_file_name = f"{saved_dir}/{dataset}_top{topK}.jsonl"

            with open(dataset_file_name, 'w', encoding="utf-8") as f_result:
                for d in _results:
                    json.dump(d, f_result)
                    f_result.write('\n')
                print(f"saved filter {dataset}: {dataset_file_name}")
            print("**************************************************************************\n")