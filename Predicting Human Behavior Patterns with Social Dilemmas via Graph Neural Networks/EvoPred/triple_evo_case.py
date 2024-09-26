# 2023/10/24 by Huaiyu Tan

import torch
import Pre_Data
import GCN_triple_case
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Parameters
    N_num = 2500
    L = 1500
    r = 1.4
    e = 0.2
    aver_k = 4
    Epoch = 2000
    Evo_size = 0.08
    output_size = 3
    hidden_size = 32
    mini_batch = None
    print("============ Evolutionary Dynamic Process ============")
    g = Pre_Data.Nets(N_num, aver_k).gen_net("SL")
    (margin_data1, T_label1) = Pre_Data.Dynamics(N_num, L, g).game_triple_dyn(r, e)
    Y = torch.Tensor(T_label1).long()
    gnn_dataset_t_lst = [
        Pre_Data.Nets(N_num, aver_k).graph_data_set(
            "comb", g, Y[:, i].reshape((N_num, 1)), margin_data1[:, i]
        ) for i in range(L)
    ]

    dataset = gnn_dataset_t_lst[int(L * Evo_size)]
    input_size = dataset.num_node_features
    # torch.manual_seed(19988449974)
    model = GCN_triple_case.GCN(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06, weight_decay=5e-4)
    print("============== Graph Convolution Network ==============")
    for epoch in range(Epoch):
        Embeddings = GCN_triple_case.train(model, dataset, optimizer)
        if epoch % 200 == 0:
            Accuracy, pre_vec = GCN_triple_case.test(model, dataset)
            print("(Test) [Epoch {}] >>Acc:{:.4f}".format(epoch, Accuracy))

    # Evolutionary Process
    Fc_real = []
    Fd_real = []
    Fe_real = []
    for i in range(L):
        x = Y[:, i].numpy().ravel()
        num_C = len(x) - np.count_nonzero(x)
        num_D = np.sum(x[np.where(x == 1)])
        num_E = np.count_nonzero(x) - num_D
        Fc_real.append(num_C / len(x))
        Fd_real.append(num_D / len(x))
        Fe_real.append(num_E / len(x))
    Fc_real = Fc_real[0: len(Fc_real) - 1]
    Fd_real = Fd_real[0: len(Fd_real) - 1]
    Fe_real = Fe_real[0: len(Fe_real) - 1]
    Fc_pre = []
    Fd_pre = []
    Fe_pre = []
    for i in range(L-1):
        dataset = gnn_dataset_t_lst[i]
        _, pre_vec = GCN_triple_case.full_test(model, dataset)
        num_C_pre = len(pre_vec) - np.count_nonzero(pre_vec)
        num_D_pre = np.sum(pre_vec[np.where(pre_vec == 1)])
        num_E_pre = np.count_nonzero(pre_vec) - num_D_pre
        Fc_pre.append(num_C_pre / len(pre_vec))
        Fd_pre.append(num_D_pre / len(pre_vec))
        Fe_pre.append(num_E_pre / len(pre_vec))
    MSE, MAE, JS_divergence, Dist_Euclidean = GCN_triple_case.evo_test(Fc_real, Fc_pre)
    print(
        "(Evo Test) MSE:{:.4f}  MAE:{:.4f}  JS:{:.4f}  DistE:{:.4f}".format(
            MSE, MAE, JS_divergence, Dist_Euclidean
        )
    )
    evo_pre = {
        'Fc_real': Fc_real, 'Fd_real': Fd_real, 'Fe_real': Fe_real, 'Fc_pre': Fc_pre, 'Fd_pre': Fd_pre, 'Fe_pre': Fe_pre
    }
    pre_perf = {'result': [MSE, MAE, JS_divergence, Dist_Euclidean]}
    pd.DataFrame(evo_pre).to_csv(
        "C:/Users/killspeeder/Desktop/PDG_triple_SL_N={}_b={}_e={}_EvoPre.csv".format(N_num, r, e), index=False
    )
    pd.DataFrame(pre_perf).to_csv(
        "C:/Users/killspeeder/Desktop/PDG_triple_SL_N={}_b={}_e={}_EvoPef.csv".format(N_num, r, e), index=False
    )
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.Figure((7, 7))
    plt.ylim(-0.02, 1)
    plt.xlabel('Rounds of Dynamic Process', fontsize=15)
    plt.ylabel('Fraction of strategies', fontsize=15)
    plt.plot(Fc_real, color="blue", label='Real Fc', marker='o')
    plt.plot(Fd_real, color="blue", label='Real Fd', marker='+')
    plt.plot(Fe_real, color="blue", label='Real Fe', marker='*')
    plt.plot(Fc_pre, color="red", label='Learned Fc', linestyle=':', marker='v')
    plt.plot(Fd_pre, color="red", label='Learned Fd', linestyle=':', marker='>')
    plt.plot(Fe_pre, color="red", label='Learned Fe', linestyle=':', marker='.')
    plt.legend(loc="upper right", fontsize=10)
    plt.show()
