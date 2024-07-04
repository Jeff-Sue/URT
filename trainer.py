import numpy as np
import torch
from tqdm import tqdm
# from sklearn.metrics import f1_score, accuracy_score
# import json


def train_or_eval_model(model, dataloader, labels, device, args, optimizer=None, train=False):
    losses = []
    assert not train or optimizer != None
    if train:  # 训练模式
        model.train()
    else: # 验证模式
        model.eval()

    # torch.autograd.set_detect_anomaly(True)
    for param in model.parameters():
        param.requires_grad = True

    num = 0
    for data in tqdm(dataloader):
        # print("data: ", data)
        label = labels[num]
        # num += 1
        max_n = 20
        mask = torch.zeros(max_n, max_n).to(device)
        for i in range(len(data) - 1):
            for j in range(i, len(data)):
                mask[i, j] = 1
        # mask[:len(data), :len(data)] = 1
        if train:
            optimizer.zero_grad()

        ### Data Format
        _,_,diff_loss = model(data, label, False)
        # print("diff_loss: ", diff_loss)
        # l1_lambda = 1e-7 # L1正则化的系数
        # l1_norm = sum(p.abs().sum() for p in model.parameters())
        # print("L1 loss: ", l1_lambda * l1_norm)
        # diff_loss += l1_lambda * l1_norm
        losses.append(diff_loss.item())

        if train:
            diff_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            with torch.no_grad():

                model.W_r_t *= mask
            # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            # 定义权重
            # weight1 = 1.0
            # weight2 = 0.5.s
            #
            # # 加权更新梯度
            # for param in model.Topmodel.parameters():
            #     if param.grad != None:
            #         param.grad *= weight1
            #
            # for param in model.GAT.parameters():
            #     if param.grad != None:
            #         param.grad *= weight1
            #
            # for param in model.rhe_model.parameters():
            #     if param.grad != None:
            #         param.grad *= weight2
            #
            # for param in model.W_r_t:
            #     if param.grad != None:
            #         param.grad *= weight2

            optimizer.step()

    avg_loss = round(np.sum(losses) / len(losses), 4)
    print("avg_loss: ", avg_loss)


    return avg_loss
