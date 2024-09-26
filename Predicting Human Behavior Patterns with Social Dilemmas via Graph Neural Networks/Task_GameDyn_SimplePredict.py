# 2023/09/28 by Huaiyu Tan

import torch
import Pre_Data
import GCN_Model

# Training Model
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

    # torch.manual_seed(19988449974)
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
            print(
                "Embedding of GCN: {}".format(Embeddings)
            )
    # _, pre_val = model(dataset)[0].max(dim=1)
    # print(pre_val[dataset.test_mask])
    # print(dataset.y.reshape(dataset.y.shape[0])[dataset.test_mask])
