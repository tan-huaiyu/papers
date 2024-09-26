# 2023/10/31 by Huaiyu Tan
# Grid-Search for parameters optimization

import torch
import Pre_Data
import GCN_Model
import numpy as np
import pandas as pd

if __name__ == "__main__":
    # Parameters
    N_num = 900
    L = 500
    save_step = int(L*0.1)
    r_lst = [i/100 for i in range(100, 201)]
    aver_k = 4
    Epoch = 2000
    Evo_size_lst = [i/100 for i in range(5, 70)]
    output_size = 2
    hidden_size = 32
    mini_batch = None

    print("============ Evolutionary Dynamic Process ============")
    g = Pre_Data.Nets(N_num, aver_k).gen_net("SL")
    b_phase = []
    evo_phase = np.zeros(len(r_lst))
    Fc_pre_phase = []
    Fc_real_phase = []
    t_flag = 0
    for r in r_lst:
        print("b = {}".format(r))
        print("================")
        print("================")
        print("================")
        (margin_data1, T_label1) = Pre_Data.Dynamics(N_num, L, g).game_dyn(r, "PDG")
        Y = torch.Tensor(T_label1).long()
        gnn_dataset_t_lst = [
            Pre_Data.Nets(N_num, aver_k).graph_data_set(
                "comb", g, Y[:, i].reshape((N_num, 1)), margin_data1[:, i]
            ) for i in range(L)
        ]

        MSE_old = 1.
        Fc_pre = None
        Fc_real = [1-np.sum(Y[:, i].numpy().ravel())/N_num for i in range(L)]
        Fc_real = Fc_real[0: len(Fc_real) - 1]
        for Evo_size in Evo_size_lst:
            print("evo size = {}".format(Evo_size))
            dataset = gnn_dataset_t_lst[int(L * Evo_size)]
            input_size = dataset.num_node_features
            # torch.manual_seed(19988449974)
            model = GCN_Model.GCN(input_size, hidden_size, output_size)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.06, weight_decay=5e-4)
            print("============== Graph Convolution Network ==============")
            for epoch in range(Epoch):
                Embeddings = GCN_Model.train(model, dataset, optimizer)
                if epoch % 1000 == 0:
                    Accuracy, F1score, TPR, TNR, FPR, FNR, pre_vec = GCN_Model.test(model, dataset)
                    print(
                        "(Test) [Epoch {}] >>Acc:{:.4f} >>F1:{:.4f} >>TPR:{:.4f} >>FPR:{:.4f}".format(
                            epoch, Accuracy, F1score, TPR, FPR
                        )
                    )

            # Evolutionary Process
            Fc_pre_temp = []
            for i in range(L-1):
                dataset = gnn_dataset_t_lst[i]
                _, _, _, _, _, _, pre_vec = GCN_Model.full_test(model, dataset)
                Fc_pre_temp.append(1 - np.sum(pre_vec) / N_num)
            MSE, MAE, _, JS_divergence, Dist_Euclidean = GCN_Model.evo_test(Fc_real, Fc_pre_temp)
            print(
                "(Evo Test) MSE:{:.4f}  MAE:{:.4f}  JS:{:.4f}  DistE:{:.4f}".format(
                    MSE, MAE, JS_divergence, Dist_Euclidean
                )
            )
            if MSE < MSE_old:
                Fc_pre = Fc_pre_temp
                evo_phase[t_flag] = Evo_size
            MSE_old = MSE
        t_flag += 1

        # Saving Results
        b_phase.append(r)
        Fc_pre_phase.append(np.average(Fc_pre[-save_step:len(Fc_pre)]))
        Fc_real_phase.append(np.average(Fc_real[-save_step:len(Fc_real)]))
    df_dic = {'b': b_phase, 'evo_size': evo_phase, 'Fc_real': Fc_real_phase, 'Fc_pre': Fc_pre_phase}
    pd.DataFrame(df_dic).to_csv("C:/Users/killspeeder/Desktop/PDG_SL_Evo_Phase_k=0.01.csv", index=False)
