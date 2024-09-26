# 2023/10/12 by Huaiyu Tan

import torch
import Pre_Data
import GCN_Model
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parameters
    N_num = 400
    L = 100
    r = 1.034
    aver_k = 4
    Epoch = 2000
    Evo_size = 0.5
    output_size = 2
    hidden_size = 32
    mini_batch = None
    print("============ Evolutionary Dynamic Process ============")
    g = Pre_Data.Nets(N_num, aver_k).gen_net("SL")
    (margin_data1, T_label1) = Pre_Data.Dynamics(N_num, L, g).game_dyn(r, "PDG")
    Y = torch.Tensor(T_label1).long()
    gnn_dataset_t_lst = [
        Pre_Data.Nets(N_num, aver_k).graph_data_set(
            "comb", g, Y[:, i].reshape((N_num, 1)), margin_data1[:, i]
        ) for i in range(L)
    ]
    dataset = gnn_dataset_t_lst[int(L * Evo_size)]
    input_size = dataset.num_node_features
    torch.manual_seed(19988449974)
    model = GCN_Model.GCN(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06, weight_decay=5e-4)
    print("============== Graph Convolution Network ==============")
    for epoch in range(Epoch):
        Embeddings = GCN_Model.train(model, dataset, optimizer)
        if epoch % 100 == 0:
            Accuracy, F1score, TPR, TNR, FPR, FNR, pre_vec = GCN_Model.test(model, dataset)
            print(
                "(Test) [Epoch {}] >>Acc:{:.4f} >>F1:{:.4f} >>TPR:{:.4f} >>FPR:{:.4f}".format(
                    epoch, Accuracy, F1score, TPR, FPR
                )
            )
        if epoch % 1000 == 0:
            print("Embedding of GCN: {}".format(Embeddings))
    # Evolutionary Process
    Evo_size_lst = [i/100+.01 for i in range(100)]
    Fc_real = [1-np.sum(Y[:, i].numpy().ravel())/N_num for i in range(L)]
    Fc_real = Fc_real[0: len(Fc_real)-1]
    Fc_pre = []
    # for i in Evo_size_lst[0: len(Evo_size_lst)-1]:
    for i in range(L-1):
        # dataset = gnn_dataset_t_lst[int(L * i)]
        dataset = gnn_dataset_t_lst[i]
        _, _, _, _, _, _, pre_vec = GCN_Model.full_test(model, dataset)
        Fc_pre.append(1 - np.sum(pre_vec) / N_num)
    MSE, MAE, KL_divergence, JS_divergence, Dist_Euclidean = GCN_Model.evo_test(Fc_real, Fc_pre)
    print(
        "(Evo Test) MSE:{:.4f}  MAE:{:.4f}  KL:{:.4f}  JS:{:.4f}  DistE:{:.4f}".format(
            MSE, MAE, KL_divergence, JS_divergence, Dist_Euclidean
        )
    )
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
