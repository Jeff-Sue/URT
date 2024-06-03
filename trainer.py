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
        max_n = 15
        mask = torch.zeros(max_n, max_n).to(device)
        for i in range(len(data) - 1):
            for j in range(i, len(data)):
                mask[i, j] = 1
        # mask[:len(data), :len(data)] = 1
        if train:
            optimizer.zero_grad()

        ### Data Format
        diff_loss = model(data, label)
        losses.append(diff_loss.item())

        if train:
            diff_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            with torch.no_grad():
                model.W_r_t *= mask

            optimizer.step()

    avg_loss = round(np.sum(losses) / len(losses), 4)
    print("avg_loss: ", avg_loss)


    return avg_loss