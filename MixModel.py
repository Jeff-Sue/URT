import torch
import torch.nn as nn
import torch.nn.functional as F
from TopModel import SegModel
from RheModel import single_execution
from transformers import BertTokenizer, AutoTokenizer, BartModel, AutoConfig
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
import copy
from eisner import eisner_matrix
from eisner import eisner
from eisner import edges_to_dg
import numpy as np
from GAT2 import GATLayerImp1
import math
from RHE_TOP import RheInfer
from TOP_RHE2 import HCExtraction, TopHC2Matrix, TopSingle2HC, compute_boundary
from TOP_RHE3 import topmatrix


class CustomLoss(nn.Module):

    def __init__(self, alpha=0.001, beta=0.001):
        super(CustomLoss, self).__init__()
        self.alpha = alpha  # 正则项的权重
        self.beta = beta  # 正则项的权重

    def kl_divergence(self, p, q, smooth=1e-9):
        p = p + smooth  # 添加平滑项
        q = q + smooth  # 添加平滑项
        return torch.sum(p * (torch.log(p) - torch.log(q)))

    def forward(self, matrix1, matrix2):
        # 计算MSE损失
        # mse_loss = nn.MSELoss()(matrix1, matrix2)

        # 计算正则项
        # 通过索引矩阵获取上半区元素
        rows1, cols1 = torch.triu_indices(matrix1.size(0), matrix1.size(1), offset=1)
        upper_half_tensor1 = matrix1[rows1, cols1]

        rows2, cols2 = torch.triu_indices(matrix2.size(0), matrix2.size(1), offset=1)
        upper_half_tensor2 = matrix2[rows2, cols2]

        mse_loss = nn.MSELoss()(upper_half_tensor1, upper_half_tensor2)
        var_matrix1 = torch.var(upper_half_tensor1)
        var_matrix2 = torch.var(upper_half_tensor2)
        reg_term1 = 1. / (var_matrix1 + 1e-8)  # 加上一个小的常数防止除以0
        reg_term2 = 1. / (var_matrix2 + 1e-8)
        kl_loss = self.kl_divergence(matrix1, matrix2)
        # 计算总损失
        # total_loss = 1000 * mse_loss + self.alpha * abs(kl_loss)
        # total_loss = 100 * mse_loss + self.alpha * (reg_term2 + reg_term1) - self.beta * (torch.mean(matrix1) + torch.mean(matrix2))
        # total_loss = 10 * mse_loss - self.alpha * (reg_term2 + reg_term1) - self.beta * (
        #             1.0 / torch.mean(matrix1) + 1.0 / torch.mean(matrix2))
        total_loss = 10 * mse_loss
        # total_loss = 100 * mse_loss

        return total_loss


class MixModel(nn.Module):
    def __init__(self, args, gnn_layers=2):
        super().__init__()
        self.args = args
        # torch.manual_seed(42)
        self.Topmodel = SegModel(args)
        self.Topmodel.to(args.device)
        self.fc1 = nn.Linear(args.emb_dim, args.hidden_dim)
        self.mse_loss = nn.MSELoss()
        self.Topmodel.load_state_dict(torch.load("model/model_doc", map_location=self.args.device), False)
        self.Topmodel.to(self.args.device)
        self.top_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        self.rhe_tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
        rhe_model_params = {"d_name": " + CNN", "name": "facebook/bart-large-cnn", "prefix_len": 1, "postfix_len": 1,
                        "max_input_size": 1024, "batch_size": 1,
                        "start_token": 0, "end_token": 2, "has_token_type_ids": False,
                        "attn_keyword": "encoder_attentions", "model": BartModel}
        torch.manual_seed(42)
        self.rhe_model = rhe_model_params["model"](AutoConfig.from_pretrained(rhe_model_params["name"],
                                                                     output_attentions=True,
                                                                     ))
        # self.RheModel = execution(rhe_model, rhe_model_params, file_path, None, layer, head, None, None, None)
        self.diff_loss = CustomLoss()

        self.affine1 = nn.Parameter(torch.empty(size=(args.hidden_dim, args.hidden_dim)))
        nn.init.xavier_uniform_(self.affine1.data, gain=1.414)

        self.drop = nn.Dropout(args.dropout)
        max_n = 20
        self.W_r_t = nn.Parameter(torch.empty(size=(max_n, max_n)))
        nn.init.xavier_uniform_(self.W_r_t.data, gain=1.414)
        # self.W_r_t = torch.nn.Parameter(torch.randn(1))
        GAT = []
        for _ in range(args.gnn_layers):
            GAT.append(GATLayerImp1(args.emb_dim, args.emb_dim, args.num_of_heads))

        self.GAT = nn.ModuleList(GAT)
        self.GAT = self.GAT.to(self.args.device)


    def kl_divergence(self, p, q, smooth=1e-9):
        p = p + smooth  # 添加平滑项
        q = q + smooth  # 添加平滑项
        return torch.sum(p * (torch.log(p) - torch.log(q)))

    def generate_list(self, n):
        result = []
        for i in range(1, n):
            for j in range(i, n):
                result.append(j)
        return result


    def one_add(self, matrix):
        one_matrix = torch.zeros_like(matrix).to(matrix.device)
        e_rows, e_cols = torch.triu_indices(matrix.size(0), matrix.size(1), offset=1)
        upper_half_e = 1 - matrix[e_rows, e_cols]
        num = 0
        for i in range(matrix.size(0)-1):
            for j in range(i+1, matrix.size(1)):
                one_matrix[i, j] = upper_half_e[num]
                num += 1

        return one_matrix


    def forward(self, input, label, PR):
        # 拿到修辞结构和放入话题的修辞结构
        rhe_matrix, e_matrix = RheInfer(self.args.device, self.rhe_tokenizer, self.rhe_model, input)

        new_matrix = torch.zeros(e_matrix.size(0), e_matrix.size(1)).to(self.args.device)
        for i in range(e_matrix.size(0)):
            new_matrix[i, i] = 1
        for i in range(e_matrix.size(0) - 1):
            for j in range(i+1, e_matrix.size(1)):
                new_matrix[i, j] = e_matrix[i, j]
                new_matrix[j, i] = e_matrix[i, j]
        new_matrix = new_matrix.clone().detach().to(self.args.device)

        # 拿到话题结构和放入修辞的话题结构

        H1, H2, C1, C2, topscores = topmatrix(self.args, self.Topmodel, input)
        top_matrix1 = torch.zeros(len(input), len(input))
        # 拿到向上拓展的top_matrix1
        for i in range(len(input)-1):
            for j in range(i+1, len(input)):
                top_matrix1[i, j] = min(topscores[i:j])
        # 拿到完全的top_matrix2
        top_matrix2 = torch.zeros_like(top_matrix1, dtype=torch.float)
        sim1_all = []
        sim2_all = []
        c_num = 0
        for i in range(top_matrix1.size(0) - 1):
            for j in range(i+1, top_matrix1.size(1)):
                sim1 = F.cosine_similarity(H1[i].unsqueeze(0), H2[j-1].unsqueeze(0), eps=1e-08)
                sim1_all.append(sim1)
                sim2 = self.Topmodel.coheren_model.cls((C1[i] + C2[j-1]) / 2.0)[0]
                c_num += 1
                sim2_all.append(sim2)
        sim1_all = torch.stack(sim1_all).squeeze(1)
        sim2_all = torch.stack(sim2_all)
        # print("sim1_all: ", sim1_all, flush=True)
        # print("sim2_all: ", sim2_all, flush=True)
        sim_all = torch.sigmoid(sim1_all + sim2_all)
        c_num = 0
        for i in range(top_matrix1.size(0) - 1):
            for j in range(i+1, top_matrix1.size(1)):
                top_matrix2[i, j] = sim_all[c_num]
                c_num += 1

        top_matrix = torch.min(top_matrix1, top_matrix2)

        # 边界值
        boundary = compute_boundary(self.args, top_matrix)
        # top_matrix = self.one_add(top_matrix)
        # boundary1 = compute_boundary(self.args, top_matrix)

        # 拿到对应topic和coherence的hidden states的修辞结构
        list_order1, list_order2 = [], []
        e_matrix_new1 = torch.zeros(H1.size(0), H1.size(0)).to(self.args.device)
        e_matrix_new2 = torch.zeros(H2.size(0), H2.size(0)).to(self.args.device)
        e_matrix_new3 = torch.zeros(C1.size(0), C1.size(0)).to(self.args.device)
        e_matrix_new4 = torch.zeros(C2.size(0), C2.size(0)).to(self.args.device)
        for l_i in range(len(input) - 1):
            for l_j in range(l_i+1, len(input)):
                # 获得【0，0，0，1，1，2】
                list_order1.append(l_i)
                # 获得【1,2,3,2,3,3】
                list_order2.append(l_j)
        for i in range(H1.size(0)-1):
            for j in range(i+1, H1.size(0)):
                if list_order1[i] == list_order1[j]:  # 相同的元素之间的关系是0
                    e_matrix_new1[i, j] = 1
                else:
                    e_matrix_new1[i, j] = e_matrix[list_order1[i], list_order1[j]]
                    e_matrix_new1[j, i] = e_matrix[list_order1[i], list_order1[j]]
        for i in range(H2.size(0)-1):
            for j in range(i+1, H2.size(0)):
                if list_order2[i] == list_order2[j]:  # 相同的元素之间的关系是0
                    e_matrix_new2[i, j] = 1
                else:
                    # print(e_matrix, flush=True)
                    # print(len(input), list_order2[i], list_order2[j], flush=True)
                    e_matrix_new2[i, j] = e_matrix[list_order2[i], list_order2[j]]
                    e_matrix_new2[j, i] = e_matrix[list_order2[i], list_order2[j]]
        e_matrix_new1 = e_matrix_new1.clone().detach().to(self.args.device)
        e_matrix_new2 = e_matrix_new2.clone().detach().to(self.args.device)
        # 获得【0，0，0，1，1，2】
        list_order3 = [i for i in range(e_matrix.size(0)-1) for _ in range(e_matrix.size(0)-1-i)]
        # 获得【1,2,3,2,3,3】
        list_order4 = self.generate_list(e_matrix.size(0))
        for i in range(C1.size(0)-1):
            for j in range(i+1, C1.size(0)):
                if list_order3[i] == list_order3[j]:  # 相同的元素之间的关系是0
                    e_matrix_new3[i, j] = 1
                else:
                    e_matrix_new3[i, j] = e_matrix[list_order3[i], list_order3[j]]
                    e_matrix_new3[j, i] = e_matrix[list_order3[i], list_order3[j]]
        for i in range(C1.size(0)-1):
            for j in range(i+1, C1.size(0)):
                if list_order4[i] == list_order4[j]:  # 相同的元素之间的关系是0
                    e_matrix_new4[i, j] = 1
                else:
                    e_matrix_new4[i, j] = e_matrix[list_order4[i], list_order4[j]]
                    e_matrix_new4[j, i] = e_matrix[list_order4[i], list_order4[j]]
        e_matrix_new3 = e_matrix_new3.clone().detach().to(self.args.device)
        e_matrix_new4 = e_matrix_new4.clone().detach().to(self.args.device)

        # 把修辞结构和话题结构放入GAT
        H_1 = [H1]
        for l in range(self.args.gnn_layers):
            H1_1 = self.GAT[l](H_1[l], new_matrix, False, e_matrix.size(0))
            H_1.append(H1_1)
        H_2 = [H2]
        for l in range(self.args.gnn_layers):
            H1_2 = self.GAT[l](H_2[l], new_matrix, False, e_matrix.size(0))
            H_2.append(H1_2)
        H_3 = [C1]
        for l in range(self.args.gnn_layers):
            H1_3 = self.GAT[l](H_3[l], new_matrix, False, e_matrix_new1.size(0))
            H_3.append(H1_3)
        H_4 = [C2]
        for l in range(self.args.gnn_layers):
            H1_4 = self.GAT[l](H_4[l], new_matrix, False, e_matrix_new2.size(0))
            H_4.append(H1_4)
        # print(H1_1.size())
        # 利用GAT得到的新的topic和coherence的hidden states得到TOP_RHE结构
        top_rhe_matrix = torch.zeros_like(rhe_matrix, dtype=torch.float, requires_grad=True)
        temp_top_rhe_matrix1 = top_rhe_matrix.clone()
        temp_top_rhe_matrix2 = top_rhe_matrix.clone()

        sim1_all = []
        sim2_all = []
        c_num = 0
        for i in range(rhe_matrix.size(0) - 1):
            for j in range(i+1, rhe_matrix.size(1)):
                sim1 = F.cosine_similarity(H1_1[i].unsqueeze(0), H1_2[j-1].unsqueeze(0),  eps=1e-08)
                sim1_all.append(sim1)
                sim2 = self.Topmodel.coheren_model.cls((H1_3[i] + H1_4[j-1]) / 2.0)[0]
                c_num += 1
                sim2_all.append(sim2)


        # 直接相加 无需其他操作
        sim1_all = torch.stack(sim1_all).squeeze(1)
        sim2_all = torch.stack(sim2_all)

        sim_all = torch.sigmoid(sim1_all + sim2_all)

        sim_all2 = []
        num = 0
        for i in range(rhe_matrix.size(0) - 1):
            for j in range(i + 1, rhe_matrix.size(1)):
                if j == i + 1:
                    sim_all2.append(sim_all[num])
                num += 1

        num = 0
        for i in range(rhe_matrix.size(0) - 1):
            for j in range(i+1, rhe_matrix.size(1)):
                temp_top_rhe_matrix2[i, j] = sim_all[num]
                temp_top_rhe_matrix1[i, j] = min(sim_all2[i:j])
                num += 1
        top_rhe_matrix = torch.min(temp_top_rhe_matrix1, temp_top_rhe_matrix2)
        ###

        # 拿到复制的话题结构，和修辞结构一起得到RHE_TOP结构
        top_matrix_copy1 = top_matrix.clone().detach().to(rhe_matrix.device)


        top_matrix_copy = torch.zeros_like(top_matrix_copy1).to(top_matrix_copy1.device)
        e_rows, e_cols = torch.triu_indices(top_matrix_copy1.size(0), top_matrix_copy1.size(1), offset=1)
        upper_half_e = top_matrix_copy1[e_rows, e_cols]
        for i in range(top_matrix_copy1.size(0) - 1):
            for j in range(i + 1, top_matrix_copy1.size(1)):
                top_matrix_copy[i, j] = (top_matrix_copy1[i, j] - upper_half_e.min()) / (upper_half_e.max() - upper_half_e.min()) + 1e-4

        rhe_top_matrix = top_matrix_copy * rhe_matrix
        ###

        # 对两个复合结构进行normalization，然后进行损失函数计算
        rhe_top_matrix_norm = torch.zeros_like(rhe_top_matrix).to(self.args.device)
        top_rhe_matrix_norm = torch.zeros_like(top_rhe_matrix).to(self.args.device)
        list1 = []
        list2 = []
        for i in range(rhe_matrix.size(0) - 1):
            for j in range(i+1, rhe_matrix.size(1)):
                list1.append(rhe_top_matrix[i, j])
                list2.append(top_rhe_matrix[i, j])
        list1 = torch.stack(list1).to(self.args.device)
        list2 = torch.stack(list2).to(self.args.device)

        if rhe_matrix.size(0) > 2:
            for i in range(rhe_matrix.size(0) - 1):
                for j in range(i+1, rhe_matrix.size(1)):
                    rhe_top_matrix_norm[i, j] = (rhe_top_matrix[i, j] - list1.min()) / (list1.max() - list1.min())
                    top_rhe_matrix_norm[i, j] = (top_rhe_matrix[i, j] - list2.min()) / (list2.max() - list2.min())
        else:
            rhe_top_matrix_norm = rhe_top_matrix
            top_rhe_matrix_norm = top_rhe_matrix

        loss = self.diff_loss(rhe_top_matrix_norm, top_rhe_matrix_norm)


        return rhe_top_matrix, top_rhe_matrix, loss










