# 2023/09/28 by Huaiyu Tan

import torch
import Pre_Data
import GCN_Model
import numpy as np
import pandas as pd


# Training Model
if __name__ == "__main__":
    # Parameters
    N_num = 400
    L = 100
    r = 1.034
    aver_k = 10
    Epoch = 2000
    Evo_size = 0.7
    output_size = 2
    hidden_size = 32
    mini_batch = None
    print("============ Evolutionary Dynamic Process ============")
    g = Pre_Data.Nets(N_num, aver_k).gen_net("ER")
    (margin_data1, T_label1) = Pre_Data.Dynamics(N_num, L, g).game_dyn(r, "PDG")
    Y = torch.Tensor(T_label1).long()
    gnn_dataset_t_lst = [
        Pre_Data.Nets(N_num, aver_k).graph_data_set(
            "comb", g, Y[:, i].reshape((N_num, 1)), margin_data1[:, i]
        ) for i in range(L)
    ]
    dataset = gnn_dataset_t_lst[int(L * Evo_size)]
    input_size = dataset.num_node_features

    # torch.manual_seed(19988449974)
    model = GCN_Model.GCN(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06, weight_decay=5e-4)
    print("============== Graph Convolution Network ==============")
    for epoch in range(Epoch):
        if epoch % 200 == 0:
            print("Epoch:{}".format(epoch))
        Embeddings = GCN_Model.train(model, dataset, optimizer)

    F1_lst = []
    Acc_lst = []
    TPR_lst = []
    FPR_lst = []
    for data_set in gnn_dataset_t_lst:
        Accuracy, F1score, TPR, TNR, FPR, FNR, pre_vec = GCN_Model.test(model, data_set)
        TPR_lst.append(TPR)
        FPR_lst.append(FPR)
        F1_lst.append(F1score)
        Acc_lst.append(Accuracy)
        print(
            "(Test) >>Acc:{:.4f} >>F1:{:.4f} >>TPR:{:.4f} >>FPR:{:.4f}".format(
                Accuracy, F1score, TPR, FPR
            )
        )
    F1_aver = np.average(np.array(F1_lst))
    F1_stds = np.std(np.array(F1_lst))
    Acc_aver = np.average(np.array(Acc_lst))
    Acc_stds = np.std(np.array(Acc_lst))
    TPR_aver = np.average(np.array(TPR_lst))
    TPR_stds = np.std(np.array(TPR_lst))
    FPR_aver = np.average(np.array(FPR_lst))
    FPR_stds = np.std(np.array(FPR_lst))
    result_aver = np.array([Acc_aver, F1_aver, TPR_aver, FPR_aver])
    result_stds = np.array([Acc_stds, F1_stds, TPR_stds, FPR_stds])
    result_train = {'aver': result_aver, 'stds': result_stds}
    pd.DataFrame(result_train).to_csv("C:/Users/killspeeder/Desktop/PDG_ER_train.csv", index=False)
