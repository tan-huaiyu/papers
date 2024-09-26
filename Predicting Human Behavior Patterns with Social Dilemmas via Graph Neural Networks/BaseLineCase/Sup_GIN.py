# 2023/11/11 by Huaiyu Tan

import torch
import Pre_Data
import numpy as np
import pandas as pd
import torch.nn.functional as nn_f
import torch_geometric.nn as pyg_nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


class GIN(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super(GIN, self).__init__()
        self.conv1 = pyg_nn.GINConv(
            torch.nn.Sequential(
                torch.nn.Linear(in_size, hid_size),
                torch.nn.ReLU(),
                torch.nn.Linear(hid_size, hid_size)
            )
        )
        self.conv2 = pyg_nn.GCNConv(hid_size, hid_size)
        self.conv3 = pyg_nn.GCNConv(hid_size, hid_size)
        self.linear = torch.nn.Linear(hid_size, out_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, my_data):
        x, edge_index = my_data.x, my_data.edge_index
        x = self.conv1(x, edge_index)
        x = nn_f.relu(x)
        x = self.conv2(x, edge_index)
        x = nn_f.dropout(x, p=0.1, training=self.training)
        x = self.conv3(x, edge_index)
        # x = pyg_nn.global_mean_pool(x, None)
        x_emb = x
        x = self.linear(x)
        return self.log_softmax(x), x_emb


def train(models, data, optimizers):
    models.train()
    optimizers.zero_grad()
    out, x_emb = models(data)
    loss = nn_f.nll_loss(out[data.train_mask], data.y.reshape(data.y.shape[0])[data.train_mask])
    loss.backward()
    optimizers.step()
    return x_emb


@torch.no_grad()
def test(models, data):
    models.eval()
    _, predicted_val = models(data)[0].max(dim=1)
    y_true = data.y[data.test_mask].numpy().ravel()
    y_pre = predicted_val[data.test_mask].numpy().ravel()
    con_mat = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TP, FP, FN, TN = con_mat[0, 0], con_mat[0, 1], con_mat[1, 0], con_mat[1, 1]
    tpr = TP / (TP + FN + .001)
    tnr = TN / (TN + FP + .001)
    fpr = FP / (FP + TN + .001)
    fnr = FN / (TP + FN + .001)
    acc = (TP + TN) / (TP + FP + FN + TN + .001)
    micro_f1 = (2 * (TP / (TP + FP + .001)) * (TP / (TP + FN + .001))) / (
            (TP / (TP + FP + .001)) + (TP / (TP + FN + .001)) + .001)
    return acc, micro_f1, tpr, tnr, fpr, fnr, predicted_val.numpy().ravel()


@torch.no_grad()
def full_test(models, data):
    models.eval()
    _, predicted_val = models(data)[0].max(dim=1)
    y_true = data.y.numpy().ravel()
    y_pre = predicted_val.numpy().ravel()
    con_mat = confusion_matrix(y_true, y_pre, labels=[0, 1])
    TP, FP, FN, TN = con_mat[0, 0], con_mat[0, 1], con_mat[1, 0], con_mat[1, 1]
    tpr = TP / (TP + FN + .001)
    tnr = TN / (TN + FP + .001)
    fpr = FP / (FP + TN + .001)
    fnr = FN / (TP + FN + .001)
    acc = (TP + TN) / (TP + FP + FN + TN + .001)
    micro_f1 = (2 * (TP / (TP + FP + .001)) * (TP / (TP + FN + .001))) / (
            (TP / (TP + FP + .001)) + (TP / (TP + FN + .001)) + .001)
    return acc, micro_f1, tpr, tnr, fpr, fnr, y_pre


def evo_test(real_data, pre_data):
    pres = np.array(pre_data)
    reals = np.array(real_data)
    MSE = mean_squared_error(reals, pres)
    MAE = mean_absolute_error(reals, pres)
    Dist_Euclidean = np.linalg.norm(reals - pres)
    KL_divergence = np.sum(np.where(reals != 0, reals * np.log(reals / pres), 0))
    JS_divergence = 0.5*np.sum(reals*np.log(reals/((reals+pres)/2)))+0.5*np.sum(pres*np.log(pres/((reals+pres)/2)))
    return MSE, MAE, KL_divergence, JS_divergence, Dist_Euclidean


if __name__ == "__main__":
    # Parameters
    N_num = 400
    L = 100
    r = 1.01
    Epoch = 10
    Evo_size = 0.2
    output_size = 2
    hidden_size = 32
    print("============ Evolutionary Dynamic Process ============")
    g = Pre_Data.Nets(N_num, 4).gen_net("SL")
    (margin_data1, T_label1) = Pre_Data.Dynamics(N_num, L, g).game_dyn(r, "PDG")
    Y = torch.Tensor(T_label1).long()
    gnn_dataset_t_lst = [
        Pre_Data.Nets(N_num, 4).graph_data_set(
            "comb", g, Y[:, i].reshape((N_num, 1)), margin_data1[:, i]
        ) for i in range(L)
    ]

    dataset_fix = gnn_dataset_t_lst[int(L*Evo_size)]
    input_size = dataset_fix.num_node_features
    model = GIN(input_size, hidden_size, output_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.06, weight_decay=5e-4)

    print("============== Graph Isomorphism Networks ==============")
    F1 = []
    ACC = []
    TPR = []
    FPR = []
    Fc_pre = []
    Accuracy, F1score, Tpr, Fpr = None, None, None, None
    Fc_real = [1 - np.sum(Y[:, i].numpy().ravel()) / N_num for i in range(L)]
    for times in range(L):
        print("# {}".format(times+1))
        dataset = gnn_dataset_t_lst[times]
        for epoch in range(Epoch):
            train(model, dataset, optimizer)
            Accuracy, F1score, Tpr, _, Fpr, _, pre_vec = test(model, dataset)
        TPR.append(Tpr)
        FPR.append(Fpr)
        F1.append(F1score)
        ACC.append(Accuracy)

    for epoch in range(Epoch):
        train(model, dataset_fix, optimizer)
    for i in range(L):
        dataset = gnn_dataset_t_lst[i]
        _, _, _, _, _, _, pre_vec = full_test(model, dataset)
        Fc_pre.append(1 - np.sum(pre_vec) / N_num)
    Mse, Mae, _, JS, Dist = evo_test(Fc_real, Fc_pre)

    # Save Results
    evo_pre = {'result': [Mse, Mae, JS, Dist]}
    pd.DataFrame(evo_pre).to_csv("C:/Users/killspeeder/Desktop/GIN_evo.csv", index=False)
    train_pre = {'ACC': ACC, 'F1': F1, 'TPR': TPR, 'FPR': FPR}
    pd.DataFrame(train_pre).to_csv("C:/Users/killspeeder/Desktop/GIN_train.csv", index=False)
