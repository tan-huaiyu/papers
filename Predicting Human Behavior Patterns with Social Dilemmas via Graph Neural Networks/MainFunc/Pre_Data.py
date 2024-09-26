# 2023/09/26 Huaiyu Tan

import torch
import random
import numpy as np
import pandas as pd
import networkx as nx
from torch.utils.data import Dataset
from torch_geometric.data import Data
# from torch.utils.data import DataLoader
# from torch.utils.data import random_split


# Build RNN DataSet(Features, Labels)
class RnnDataSet(Dataset):
    def __init__(self, DATA: torch.Tensor, LABELS: torch.Tensor):
        self.dataset = DATA  # dim(N, L, N)
        self.labels = LABELS  # dim(N, L)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx], self.labels[idx]


# T_M_I_F_E
def t_m_i_f_e(A_mat, marg_t):
    n = len(marg_t)
    Delta_mat = np.zeros([n, n])
    for i in range(n):
        for j in range(n):
            Delta_mat[i, j] = np.abs(marg_t[i]-marg_t[j])
    return np.multiply(A_mat, Delta_mat)


# Networks Structure & Build GNN DataLoader
class Nets:
    def __init__(self, N: int, deg_k: int):
        self.N = N
        self.deg = deg_k

    def gen_net(self, net_type: str):
        if net_type == "SL":
            n = int(np.sqrt(self.N))
            G_temp = nx.grid_2d_graph(n, n, periodic=True)
            G = nx.convert_node_labels_to_integers(G_temp)
        elif net_type == "BASF":
            G = nx.barabasi_albert_graph(self.N, int(self.deg/2))
        else:
            G = nx.erdos_renyi_graph(self.N, self.deg/(self.N-1))
        return G

    def sample_mask(self, idx):
        mask = torch.zeros(self.N)
        mask[idx] = 1
        return torch.as_tensor(mask, dtype=torch.bool)

    # 特征编码方式：默认用邻接矩阵、T_M_I_F_E、comb
    def graph_data_set(self, Feature_type, G: nx.Graph, Label_t, Margin_t):
        edge_lst = []
        for i in range(self.N):
            for j in G.edges(i):
                edge_lst.append(list(j))
        edge_index = torch.tensor(edge_lst, dtype=torch.long)
        if Feature_type == "comb":
            Adj = nx.to_numpy_array(G)
            Mar_adj = np.c_[Adj, Margin_t]
            x = torch.tensor(Mar_adj, dtype=torch.float32)
        elif Feature_type == "T_M_I_F_E":
            Adj = nx.to_numpy_array(G)
            Adj_t_m_i_f_e = t_m_i_f_e(Adj, Margin_t)
            Mar_adj = np.c_[Adj_t_m_i_f_e, Margin_t]
            x = torch.tensor(Mar_adj, dtype=torch.float32)
        else:
            x = torch.tensor(nx.to_numpy_array(G), dtype=torch.float32)
        train_idx = range(int(0.7 * self.N))
        test_idx = range(int(0.7 * self.N), int(self.N))
        train_mask = Nets.sample_mask(self, train_idx)
        test_mask = Nets.sample_mask(self, test_idx)
        gnn_data = Data(
            x=x, y=Label_t, edge_index=edge_index.t().contiguous(), train_mask=train_mask, test_mask=test_mask
        )
        return gnn_data


# Game Structure
class Game:
    def __init__(self, factor):
        self.factor = factor
        self.C = np.array([1, 0])
        self.D = np.array([0, 1])
        self.C2 = np.array([1, 0, 0])
        self.D2 = np.array([0, 1, 0])
        self.E2 = np.array([0, 0, 1])

    def pdg(self):  # b in [1, 2]
        m_pay = np.array([[1, 0], [self.factor, 0]])
        return m_pay

    def sdg(self):  # b in [1, 2]
        m_pay = np.array([[1, 0], [self.factor, -1]])
        return m_pay

    def hg(self):  # b in [0, 1]
        m_pay = np.array([[1, 0], [self.factor, -1]])
        return m_pay

    def shg(self):  # b in [0, 1]
        m_pay = np.array([[1, -1], [self.factor, 0]])
        return m_pay

    def exit_game(self, e):
        m_pay = np.array([[1, 0, 0], [self.factor, 0, 0], [e, e, e]])
        return m_pay

    def payoff(self, si, m_pay, sj):
        if si == 0:
            s_i = self.C
        else:
            s_i = self.D
        if sj == 0:
            s_j = self.C
        else:
            s_j = self.D
        phi = np.matmul(np.matmul(s_i, m_pay), s_j.T)
        return phi

    def payoff2(self, si, m_pay, sj):
        if si == 0:
            s_i = self.C2
        elif si == 1:
            s_i = self.D2
        else:
            s_i = self.E2
        if sj == 0:
            s_j = self.C2
        elif sj == 1:
            s_j = self.D2
        else:
            s_j = self.E2
        phi = np.matmul(np.matmul(s_i, m_pay), s_j.T)
        return phi


# Dynamic Process
class Dynamics:
    def __init__(self, N, Rounds, G):
        self.N = N
        self.G = G
        self.Rounds = Rounds

    # Evolutionary Game Dynamic
    def game_dyn(self, game_para, game_type):
        # Initialize Strategies Set
        if game_type == "PDG":
            M = Game(game_para).pdg()
        elif game_type == "SDG":
            M = Game(game_para).sdg()
        elif game_type == "HG":
            M = Game(game_para).hg()
        else:
            M = Game(game_para).shg()
        Real_Label = np.zeros((self.N, self.Rounds))
        Margin_Tensor = np.zeros((self.N, self.Rounds))
        s_set = [random.randint(0, 1) for _ in range(self.N)]
        for t in range(self.Rounds):
            Real_Label[:, t] = np.array(s_set)
            if t % 100 == 0:
                print(f"Rounds: {t}")
            # Game process
            for i in range(self.N):
                for j in list(self.G.neighbors(i)):
                    Margin_Tensor[i, t] += Game(None).payoff(s_set[i], M, s_set[j])
            # Synchronous Updating
            s_set_old = [v for v in s_set]
            for i in range(self.N):
                if list(self.G.neighbors(i)):
                    rand_v = random.choice(list(self.G.neighbors(i)))
                    w = 1. / (1. + np.exp((Margin_Tensor[i, t] - Margin_Tensor[rand_v, t]) / 0.1))
                    if w > random.random():
                        s_set[i] = s_set_old[rand_v]
            # Mutation
            # for i in range(self.N):
            #     if 1e-4 > random.random():
            #         if s_set[i] == 0:
            #             s_set[i] = 1
            #         else:
            #             s_set[i] = 0
        return (
            Margin_Tensor + np.random.normal(0, .01, (self.N, self.Rounds)),
            Real_Label
        )

    # Evolutionary Game Dynamic
    def game_triple_dyn(self, game_para, e):
        # Initialize Strategies Set
        M = Game(game_para).exit_game(e)
        Real_Label = np.zeros((self.N, self.Rounds))
        Margin_Tensor = np.zeros((self.N, self.Rounds))
        s_set = [random.randint(0, 2) for _ in range(self.N)]
        for t in range(self.Rounds):
            Real_Label[:, t] = np.array(s_set)
            if t % 100 == 0:
                print(f"Rounds: {t}")
            # Game process
            for i in range(self.N):
                for j in list(self.G.neighbors(i)):
                    Margin_Tensor[i, t] += Game(None).payoff2(s_set[i], M, s_set[j])
            # Synchronous Updating
            s_set_old = [v for v in s_set]
            for i in range(self.N):
                if list(self.G.neighbors(i)):
                    rand_v = random.choice(list(self.G.neighbors(i)))
                    # Fermi updating
                    w = 1. / (1. + np.exp((Margin_Tensor[i, t] - Margin_Tensor[rand_v, t]) / 0.1))
                    if w > random.random():
                        s_set[i] = s_set_old[rand_v]
            # Mutation
            # for i in range(self.N):
            #     if 1e-3 > random.random():
            #         if s_set[i] == 0:
            #             s_set[i] = random.choice([1, 2])
            #         elif s_set[i] == 1:
            #             s_set[i] = random.choice([0, 2])
            #         else:
            #             s_set[i] = random.choice([0, 1])
        return (
            Margin_Tensor + np.random.normal(0, .01, (self.N, self.Rounds)),
            Real_Label
        )

    # Kura Coupling Dynamic
    def coupling_dyn(self, diff_h, coup_factor):
        G_adj = coup_factor*nx.to_numpy_array(self.G)
        Phase_Tensor = np.zeros((self.N, self.Rounds))
        Margin_Tensor = np.zeros((self.N, self.Rounds))
        nat_feq = np.array([random.uniform(0, 1) for _ in range(self.N)])
        init_phase = np.array([random.uniform(0, 2*np.pi) for _ in range(self.N)])
        for t in range(self.Rounds):
            print(f"Rounds: {t+1}")
            Phase_Tensor[:, t] = init_phase
            old_phase = init_phase
            for i in range(self.N):
                init_phase[i] += diff_h*(nat_feq[i]+np.dot(np.array(G_adj[i]), np.sin(init_phase-init_phase[i])))
            Margin_Tensor[:, t] = (np.array(init_phase)-np.array(old_phase))/diff_h
        return (
            Margin_Tensor + np.random.normal(0, .001, (self.N, self.Rounds)),
            Phase_Tensor
        )

    # Epidemic SIS Dynamic (Edge Infection)
    def sis_dyn(self, init_beta, init_mu, init_I):
        # Initialize S-I Set
        node_lst = list(self.G.nodes())
        Real_Label = np.zeros((self.N, self.Rounds))
        Margin_Tensor = np.zeros((self.N, self.Rounds))
        s_set = [0]*self.N
        I_lst = random.sample(node_lst, int(init_I*self.N))
        for i in I_lst:
            s_set[i] = 1
        for t in range(self.Rounds):
            Real_Label[:, t] = np.array(s_set)
            if t % 100 == 0:
                print(f"Rounds: {t+1}")
            # Synchronous Updating
            temp_beta_i = np.zeros(self.N)
            for i in range(self.N):
                neb_lst = list(self.G.neighbors(i))
                count_temp = 0
                for j in neb_lst:
                    count_temp += s_set[j]
                if s_set[i] == 0:
                    beta_i = 1. - (1.-init_beta) ** count_temp
                    temp_beta_i[i] = beta_i
                    if random.random() < beta_i:
                        s_set[i] = 1
                else:
                    temp_beta_i[i] = 0
                    if random.random() < init_mu:
                        s_set[i] = 0
            Margin_Tensor[:, t] = temp_beta_i
        return (
            Margin_Tensor + np.random.normal(0, .01, (self.N, self.Rounds)),
            Real_Label
        )


# Empirical analysis of experimental data
def real_data_test():
    margin_csv = pd.read_csv("ExperimentData/MarginMat.csv", header=None)
    choice_csv = pd.read_csv("ExperimentData/ChoiceMat.csv", header=None)
    Margin_Tensor = np.array(margin_csv)
    Real_Label = np.array(choice_csv)
    return (
        Margin_Tensor + np.random.normal(0, .01, (Margin_Tensor.shape[0], Margin_Tensor.shape[1])),
        Real_Label
    )


if __name__ == "__main__":
    N_num = 10000
    L = 200
    r = 1.034
    aver_k = 6
    mini_batch = 20
    train_ratio = 0.8
    train_num = int(N_num * train_ratio)
    test_num = int(N_num-train_num)
    g = Nets(N_num, aver_k).gen_net("SL")

    # Dynamic Data and <RNN>
    (margin_data1, T_label1) = Dynamics(N_num, L, g).game_dyn(r, "PDG")
    # X_margin = torch.Tensor(margin_data1).float()
    # X_full = torch.zeros((N_num, L, N_num))
    # for dim1 in range(X_full.shape[0]):
    #     for dim2 in range(X_full.shape[1]):
    #         X_full[dim1][dim2] = torch.Tensor(full_data1[dim2][dim1]).float()
    Y = torch.Tensor(T_label1).long()
    # from matplotlib import pyplot as plt
    # Fc = 1 - Y.sum(dim=0) / N_num
    # plt.plot(Fc)
    # rnn_data_set = RnnDataSet(X_full, Y)
    # trains, val, tests = random_split(rnn_data_set, [train_num, 0, test_num])
    # rnn_full_loader = DataLoader(rnn_data_set, batch_size=mini_batch, shuffle=True)
    # rnn_train_loader = DataLoader(trains, batch_size=10, shuffle=True)
    # rnn_test_loader = DataLoader(tests, shuffle=False)
    # Test for Loader Function
    # for batch_x, batch_y in rnn_full_loader:
    #     print(batch_x.size(), batch_y.size())

    # Network Data and <GNN>
    gnn_dataset_t_lst = [
        Nets(N_num, aver_k).graph_data_set("Adj", g, Y[:, i].reshape((N_num, 1)), None) for i in range(N_num)
    ]
    # (margin_data2, Evo_Phase2) = Dynamics(100, 200, g).coupling_dyn(0.001, 20)
    # gnn_coupling_loader = Nets(100, 6).graph_data_loader(g, Evo_Phase2)
