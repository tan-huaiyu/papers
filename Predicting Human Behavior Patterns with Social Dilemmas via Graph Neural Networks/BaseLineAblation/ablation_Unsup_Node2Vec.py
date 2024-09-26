# 2023/11/11 by Huaiyu Tan
import random
import Pre_Data
import numpy as np
import pandas as pd
import networkx as nx
from gensim.models import Word2Vec
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


class Node2Vec:
    def __init__(self, G: nx.Graph, p, q, Margin_payoffs):
        self.G = G
        self.p = p
        self.q = q
        self.Margin_Payoffs = Margin_payoffs

    # p bigger and q smaller: DFS / q bigger and p smaller: BFS / p=q=1: Deep Walk
    def sample_strategy(self, v, t):
        neb_v = list(self.G.neighbors(v))
        if len(neb_v) == 0:
            return False
        weights = [1]*len(neb_v)  # with attributes
        for i, x in enumerate(neb_v):
            if t == x:  # go back
                weights[i] = 1/self.p
            elif not self.G.has_edge(t, x):  # go forward
                weights[i] = 1/self.q
        return random.choices(neb_v, weights=weights, k=1)[0]

    def bias_walk(self, v, path_length):
        bias_sample = [v]
        neb_lst = list(self.G.neighbors(v))
        if len(neb_lst) > 0:
            bias_sample.append(random.choice(neb_lst))
            for _ in range(2, path_length):
                next_v = Node2Vec(self.G, self.p, self.q, self.Margin_Payoffs).sample_strategy(
                    bias_sample[-1], bias_sample[-2]
                )
                if not next_v:
                    break
                bias_sample.append(next_v)
        return bias_sample

    # Generate Dataset
    def get_data(self, path_len, gamma):
        # Get sequences sets
        node2vec_walks = []
        for v in list(self.G.nodes()):
            for _ in range(gamma):
                node2vec_walks.append(Node2Vec(self.G, self.p, self.q, self.Margin_Payoffs).bias_walk(v, path_len))
        return node2vec_walks


# Skip-gram structure
def node2vec_train(vec_size, sequences):
    # Model training
    model = Word2Vec(
        vector_size=vec_size,
        window=5,
        sg=1,  # 1:skip-grim / 0:C-BOW
        hs=0,  # non-hierarchical softmax
        negative=20,  # negative sampling, default = [5~20]
        alpha=.03,  # learning rate
        min_alpha=.0007,  # minimum learning rate
        workers=8,  # multi-process
        compute_loss=False,  # save the training loss
        seed=123
    )
    model.build_vocab(sequences, progress_per=2)
    model.train(sequences, total_examples=model.corpus_count, epochs=50, report_delay=1)
    return model


def full_test(y_true, y_pre):
    con_mat = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TP, FP, FN, TN = con_mat[0, 0], con_mat[0, 1], con_mat[1, 0], con_mat[1, 1]
    tpr = TP / (TP + FN + .001)
    tnr = TN / (TN + FP + .001)
    fpr = FP / (FP + TN + .001)
    fnr = FN / (TP + FN + .001)
    acc = (TP + TN) / (TP + FP + FN + TN + .001)
    micro_f1 = (2 * (TP / (TP + FP + .001)) * (TP / (TP + FN + .001))) / (
            (TP / (TP + FP + .001)) + (TP / (TP + FN + .001)) + .001)
    return acc, micro_f1, tpr, tnr, fpr, fnr


def evo_test(Real_data, Pre_data):
    pres = np.array(Pre_data)
    reals = np.array(Real_data)
    MSE = mean_squared_error(reals, pres)
    MAE = mean_absolute_error(reals, pres)
    Dist_Euclidean = np.linalg.norm(reals - pres)
    KL_divergence = np.sum(np.where(reals != 0, reals * np.log(reals / pres), 0))
    JS_divergence = 0.5*np.sum(reals*np.log(reals/((reals+pres)/2)))+0.5*np.sum(pres*np.log(pres/((reals+pres)/2)))
    return MSE, MAE, KL_divergence, JS_divergence, Dist_Euclidean


if __name__ == "__main__":
    N = 400
    L = 100
    b = 1.01
    vec_dim = 16  # Embedding dimension
    path_size = 30  # length of each sequences
    gm = 30  # number of random walk sequences for each nodes
    p_model, q_model = 1, 1  # DeepWalk
    output_size = 2
    hidden_size = 32
    g = Pre_Data.Nets(N, 4).gen_net("SL")

    # Dynamic Data
    (margin_data1, T_label1) = Pre_Data.Dynamics(N, L, g).game_dyn(b, "PDG")

    # Node2Vec
    F1 = []
    ACC = []
    TPR = []
    FPR = []
    Fc_pre = []
    Fc_real = [1 - np.sum(T_label1[:, i]) / N for i in range(L)]
    for times in range(L):
        print("# {}".format(times+1))
        print("========Generate node dictionary========")
        dataset = Node2Vec(g, p_model, q_model, margin_data1[:, times]).get_data(path_size, gm)
        print("========Train Model========")
        model_nv = node2vec_train(vec_dim, dataset)
        embedding_vec = model_nv.wv.vectors
        embedding_idx = [model_nv.wv.key_to_index[i] for i in list(g.nodes)]

        # classification by K-means
        model_k_means = KMeans(n_clusters=2, n_init='auto')
        fit_cluster = model_k_means.fit(embedding_vec)
        cluster_labels = fit_cluster.labels_
        pre_data = [cluster_labels[i] for i in embedding_idx]
        Fc_pre.append(1 - np.sum(pre_data) / N)
        # pre_data2 = []
        # for i in range(len(pre_data)):
        #     if pre_data[i] == 0:
        #         pre_data2.append(1)
        #     else:
        #         pre_data2.append(0)
        real_data = T_label1[:, times]
        Acc, f1, Tpr, _, Fpr, _ = full_test(real_data, pre_data)
        F1.append(f1)
        ACC.append(Acc)
        TPR.append(Tpr)
        FPR.append(Fpr)
    Mse, Mae, _, JS, Dist = evo_test(Fc_real, Fc_pre)

    # Save Results
    evo_pre = {'result': [Mse, Mae, JS, Dist]}
    pd.DataFrame(evo_pre).to_csv("C:/Users/killspeeder/Desktop/abl_node2vec_evo.csv", index=False)
    train_pre = {'ACC': ACC, 'F1': F1, 'TPR': TPR, 'FPR': FPR}
    pd.DataFrame(train_pre).to_csv("C:/Users/killspeeder/Desktop/abl_node2vec_train.csv", index=False)
