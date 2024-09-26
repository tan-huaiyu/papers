# 2023/09/28 by Huaiyu Tan

import torch
import numpy as np
import torch.nn.functional as nn_f
import torch_geometric.nn as pyg_nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class GCN(torch.nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.conv1 = pyg_nn.GCNConv(in_size, hid_size)
        self.conv2 = pyg_nn.GCNConv(hid_size, hid_size)
        self.linear = torch.nn.Linear(hid_size, out_size)
        self.log_softmax = torch.nn.LogSoftmax(dim=1)

    def forward(self, my_data):
        x, edge_index = my_data.x, my_data.edge_index
        x = self.conv1(x, edge_index)
        x = nn_f.relu(x)
        x = nn_f.dropout(x, p=0.1, training=self.training)
        x = self.conv2(x, edge_index)
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
    con_mat = confusion_matrix(y_true, y_pre, labels=[0, 1, 2])
    acc = np.sum(np.diag(con_mat)) / np.sum(con_mat)
    return acc, predicted_val.numpy().ravel()


@torch.no_grad()
def full_test(models, data):
    models.eval()
    _, predicted_val = models(data)[0].max(dim=1)
    y_true = data.y.numpy().ravel()
    y_pre = predicted_val.numpy().ravel()
    con_mat = confusion_matrix(y_true, y_pre, labels=[0, 1, 2])
    acc = np.sum(np.diag(con_mat)) / np.sum(con_mat)
    return acc, y_pre


def evo_test(real_data, pre_data):
    pres = np.array(pre_data)
    reals = np.array(real_data)
    MSE = mean_squared_error(reals, pres)
    MAE = mean_absolute_error(reals, pres)
    Dist_Euclidean = np.linalg.norm(reals - pres)
    JS_divergence = 0.5*np.sum(reals*np.log(reals/((reals+pres)/2)))+0.5*np.sum(pres*np.log(pres/((reals+pres)/2)))
    return MSE, MAE, JS_divergence, Dist_Euclidean
