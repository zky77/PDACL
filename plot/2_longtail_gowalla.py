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
    # 长尾推荐_Gowalla
    x = np.arange(6)
    x_label = ['0-25', '25-30', '30-40', '40-50', '50-60', '60-70']
    GNNEC_recall = [0.2395, 0.2443, 0.2314, 0.2187, 0.2005, 0.1843]
    GNNEC_ndcg = [0.1405, 0.1464, 0.1582, 0.1633, 0.1631, 0.1531]
    SGL_recall = [0.2214, 0.2247, 0.2079, 0.2028, 0.1874, 0.1660]
    SGL_ndcg = [0.1264, 0.1307, 0.1400, 0.1489, 0.1536, 0.1367]
    LightGCN_recall = [0.2128, 0.2158, 0.1952, 0.1980, 0.1822, 0.1734]
    LightGCN_ndcg = [0.1206, 0.1253, 0.1314, 0.1438, 0.1428, 0.1388]
    # SGL_recall = [0.2244,0.2277,0.2079,0.2008,0.1844,0.1560]
    # SGL_ndcg = [0.1294,0.1367,0.1413,0.1469,0.1506,0.1277]
    # LightGCN_recall = [0.2158,0.2188,0.1952,0.1960,0.1792,0.1634]
    # LightGCN_ndcg = [0.1226,0.1283,0.1314,0.1418,0.1398,0.1308]

    recall_diff = [(GNNEC_recall[i] - SGL_recall[i]) / SGL_recall[i] * 100 for i in range(len(GNNEC_recall))]
    ndcg_diff = [(GNNEC_ndcg[i] - SGL_ndcg[i]) / SGL_ndcg[i] * 100 for i in range(len(GNNEC_ndcg))]

    # width
    total_width, n = 0.8, 3
    width = total_width / n
    x = x - (total_width - width) / 2
    # plt
    plt.ylim((0.14, 0.27))
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

    plt.savefig("Recall@20 on Gowalla.png", dpi=300)
    plt.show()
    # plt
    plt.ylim((0.085, 0.17))
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

    plt.savefig("NDCG@20 on Gowalla.png", dpi=300)
    plt.show()