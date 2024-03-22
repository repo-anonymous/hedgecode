## Dataset

> [CodeSearchNet]([github/CodeSearchNet: Datasets, tools, and benchmarks for representation learning of code.](https://github.com/github/CodeSearchNet/tree/master)) is a widely used benchmark for code search task. It contains six different programming languages. Following previous works, we choose CodeSearchNet to train and evaluate HedgeCode. 
>
> We construct a relevance detection dataset for CTRD task based on the CodeSearchNet.

##### CodeSearchNet Benchmark

> You can download the dataset from [CodeSearchNet]([github/CodeSearchNet: Datasets, tools, and benchmarks for representation learning of code.](https://github.com/github/CodeSearchNet/tree/master)) or execute the following script to build the codesearchnet dataset.

~~~bash
cd ./dataset/codesearchnet 
bash build_data.sh
~~~

##### Relevance Detection Datasets 

> First, download the Cleaned CodeSearchNet Benchmark from [Geogle Drive](https://drive.google.com/file/d/1rd2Tc6oUWBo7JouwexW3ksQ0PaOhUr6h/view). This dataset combines the queries of the test and validation datasets with the ground-truth code snippets in the codebase. And store the data set in the "pairs" folder.
> 
> Then execute the following scripts to build the relevance detection datasets.

~~~bash
python build_detection_pairs.py

python data_filter.py

python combination_pairs.py
~~~

