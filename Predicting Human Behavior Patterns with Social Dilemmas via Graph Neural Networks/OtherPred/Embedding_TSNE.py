# 2023/10/24 by Huaiyu Tan

import torch
import Pre_Data
import GCN_Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

if __name__ == "__main__":
    # Parameters
    N_num = 1225
    L = 1000
    r = 1.01
    aver_k = 4
    Epoch = 2000
    Evo_size = 0.2
    output_size = 2
    hidden_size = 32
    mini_batch = None
    print("============ Evolutionary Dynamic Process ============")
    g = Pre_Data.Nets(N_num, aver_k).gen_net("SL")
    (margin_data1, T_label1) = Pre_Data.Dynamics(N_num, L, g).game_dyn(r, "PDG")
    Y = torch.Tensor(T_label1).long()
    gnn_dataset_t_lst_1 = [
        Pre_Data.Nets(N_num, aver_k).graph_data_set(
            "comb", g, Y[:, i].reshape((N_num, 1)), margin_data1[:, i]
        ) for i in range(L)
    ]

    gnn_dataset_t_lst_2 = [
        Pre_Data.Nets(N_num, aver_k).graph_data_set(
            "adj", g, Y[:, i].reshape((N_num, 1)), margin_data1[:, i]
        ) for i in range(L)
    ]

    dataset_1 = gnn_dataset_t_lst_1[int(L * Evo_size)]
    dataset_2 = gnn_dataset_t_lst_2[int(L * Evo_size)]
    input_size_1 = dataset_1.num_node_features
    input_size_2 = dataset_2.num_node_features
    # torch.manual_seed(19988449974)
    model_1 = GCN_Model.GCN(input_size_1, hidden_size, output_size)
    model_2 = GCN_Model.GCN(input_size_2, hidden_size, output_size)
    optimizer_1 = torch.optim.Adam(model_1.parameters(), lr=0.06, weight_decay=5e-4)
    optimizer_2 = torch.optim.Adam(model_2.parameters(), lr=0.06, weight_decay=5e-4)
    print("============== Graph Convolution Network ==============")
    Embeddings_1 = None
    Embeddings_2 = None
    for epoch in range(Epoch):
        Embeddings_1 = GCN_Model.train(model_1, dataset_1, optimizer_1)
        Embeddings_2 = GCN_Model.train(model_2, dataset_2, optimizer_2)
        if epoch % 200 == 0:
            Accuracy, F1score, TPR, TNR, FPR, FNR, pre_vec = GCN_Model.test(model_1, dataset_1)
            print(
                "(Test) [Epoch {}] >>Acc:{:.4f} >>F1:{:.4f} >>TPR:{:.4f} >>FPR:{:.4f}".format(
                    epoch, Accuracy, F1score, TPR, FPR
                )
            )
    emb_1 = Embeddings_1.detach().numpy()
    emb_2 = Embeddings_2.detach().numpy()
    pd.DataFrame(emb_1).to_csv("C:/Users/killspeeder/Desktop/PDG_SL_emb_comb.csv")
    pd.DataFrame(emb_2).to_csv("C:/Users/killspeeder/Desktop/PDG_SL_emb_adj.csv")

    # Evolutionary Process
    Fc_real = [1-np.sum(Y[:, i].numpy().ravel())/N_num for i in range(L)]
    Fc_real = Fc_real[0: len(Fc_real)-1]
    Fc_pre = []
    for i in range(L-1):
        # dataset = gnn_dataset_t_lst[int(L * i)]
        dataset = gnn_dataset_t_lst_1[i]
        _, _, _, _, _, _, pre_vec = GCN_Model.full_test(model_1, dataset)
        Fc_pre.append(1 - np.sum(pre_vec) / N_num)
    MSE, MAE, _, JS_divergence, Dist_Euclidean = GCN_Model.evo_test(Fc_real, Fc_pre)
    print(
        "(Evo Test) MSE:{:.4f}  MAE:{:.4f}  JS:{:.4f}  DistE:{:.4f}".format(
            MSE, MAE, JS_divergence, Dist_Euclidean
        )
    )
    # evo_pre = {'Fc_real': Fc_real, 'Fc_pre': Fc_pre}
    # pre_perf = {'result': [MSE, MAE, JS_divergence, Dist_Euclidean]}
    # pd.DataFrame(evo_pre).to_csv(
    #     "C:/Users/killspeeder/Desktop/PDG_BASF_k={}_Evo={}_EvoPre.csv".format(aver_k, Evo_size), index=False
    # )
    # pd.DataFrame(pre_perf).to_csv(
    #     "C:/Users/killspeeder/Desktop/PDG_BASF_k={}_Evo={}_EvoPef.csv".format(aver_k, Evo_size), index=False
    # )

    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.Figure((7, 7))
    plt.ylim(0, 1)
    plt.xlabel('Rounds of Dynamic Process', fontsize=15)
    plt.ylabel('Fc', fontsize=15)
    plt.plot(Fc_real, color="blue", label='Real Fc', marker='o')
    plt.plot(Fc_pre, color="red", label='Learned Fc', linestyle=':', marker='>')
    plt.legend(loc="upper right", fontsize=15)
    plt.show()

    # Visualization through T-SNE
    model_tsne = TSNE(n_components=2, n_iter=1000)
    embedding_2d_1 = model_tsne.fit_transform(emb_1)
    embedding_2d_2 = model_tsne.fit_transform(emb_2)
    plt.figure(figsize=(8, 6))
    plt.scatter(embedding_2d_2[:, 0], embedding_2d_2[:, 1])
    plt.show()
