import time
from sklearn.neighbors import BallTree
import numpy as np

def ball_tree(search_space, topK=2, query_set=None):

    print("--------------------------------star build ball tree--------------------------------------")
    start_time = time.time()

    tree = BallTree(search_space)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print("build time：{:.2f} min".format(elapsed_minutes))
    print("----------------------------------build finish---------------------------------------------")

    print("------------------------------------star search--------------------------------------------")
    start_time = time.time()

    dist, ind = tree.query(np.array(query_set), k=topK)

    end_time = time.time()
    elapsed_time = end_time - start_time
    elapsed_minutes = elapsed_time / 60
    print("search time：{:.2f} min".format(elapsed_minutes))
    print("-------------------------------------search finish----------------------------------------")

    return dist, ind