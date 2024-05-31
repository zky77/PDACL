import os
import random
import numpy as np
import seaborn as sns
import scipy.io as sio
import matplotlib.pyplot as plt
from matplotlib import gridspec
from scipy.stats import gaussian_kde
from sklearn.manifold import TSNE
import warnings
warnings.filterwarnings("ignore")
# seed
SEED = 42
os.environ['TF_DETERMINISTIC_OPS'] ='1'
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


if __name__ == '__main__':
    # 长尾推荐_Amazonbook
    x = np.arange(5)
    x_label = ['0-35', '35-40', '40-50', '50-60', '60-70']
    GNNEC_recall = [0.1432, 0.1475, 0.1398, 0.1379, 0.1506]
    GNNEC_ndcg = [0.0990, 0.1022, 0.0975, 0.0979, 0.1022]
    SGL_recall = [0.1267, 0.1136, 0.1194, 0.1200, 0.1332]
    SGL_ndcg = [0.0901, 0.0799, 0.0849, 0.0825, 0.0893]
    LightGCN_recall = [0.1130, 0.0955, 0.1003, 0.0940, 0.1077]
    LightGCN_ndcg = [0.0751, 0.0685, 0.0686, 0.0653, 0.0721]

    recall_diff = [(GNNEC_recall[i] - SGL_recall[i]) / SGL_recall[i] * 100 for i in range(len(GNNEC_recall))]
    ndcg_diff = [(GNNEC_ndcg[i] - SGL_ndcg[i]) / SGL_ndcg[i] * 100 for i in range(len(GNNEC_ndcg))]

    # width
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    # plt
    plt.ylim((0.06, 0.16))
    plt.xticks(x, x_label)
    plt.bar(x - width, GNNEC_recall, width=width, label='GNN-EC')
    plt.bar(x, SGL_recall, width=width, label='SGL')
    plt.bar(x + width, LightGCN_recall, width=width, label='LightGCN')
    plt.ylabel('Recall@20')
    plt.xlabel('Group user')
    plt.legend(loc='upper left')

    ax2 = plt.twinx()
    ax2.set_ylabel("Improve %")
    ax2.set_ylim([0, 20])
    plt.plot(x, recall_diff, "r", marker='.', c='black', ms=8, linewidth='1', label="GNN-EC diff SGL")
    plt.legend(loc='upper right')

    plt.savefig("Recall@20 on Amazon-Book.png", dpi=300)
    plt.show()
    # plt
    plt.ylim((0.04, 0.12))
    plt.xticks(x, x_label)
    plt.bar(x - width, GNNEC_ndcg, width=width, label='GNN-EC')
    plt.bar(x, SGL_ndcg, width=width, label='SGL')
    plt.bar(x + width, LightGCN_ndcg, width=width, label='LightGCN')
    plt.ylabel('NDCG@20')
    plt.xlabel('Group user')
    plt.legend(loc='upper left')

    ax2 = plt.twinx()
    ax2.set_ylabel("Improve %")
    ax2.set_ylim([0, 20])
    plt.plot(x, ndcg_diff, "r", marker='.', c='black', ms=8, linewidth='1', label="GNN-EC diff SGL")
    plt.legend(loc='upper right')

    plt.savefig("NDCG@20 on Amazon-Book.png", dpi=300)
    plt.show()