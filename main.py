import argparse
import json
import math
import random
import time
from trainer import train_or_eval_model
from torch.utils.data import DataLoader, random_split
from dataset import get_rhe_datasets_stac, get_top_datasets1, get_rhe_datasets_molweni, get_top_dataset2
from transformers import AutoTokenizer, AdamW, AutoConfig, BartModel
import torch
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from MixModel import MixModel
from RHEtest2 import compute_f1, compute_f2
import copy
from eda import eda
from topt import TopTestInfer1, TopTestInfer2, TopTestInfer_tiage
import os
from ablation_ana import analysis1, analysis2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    ### Add Parameters
    # parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", default='STAC')
    parser.add_argument("--save_name", default='epoch')
    parser.add_argument("--single_ckpt", action='store_true')
    parser.add_argument("--ckpt")
    parser.add_argument("--root", default='.')
    parser.add_argument("--no_cuda", action='store_true')
    parser.add_argument("--ckpt_start", type=int, default=0)
    parser.add_argument("--ckpt_end", type=int, default=3)

    parser.add_argument("--pick_num", type=int, default=4)
    parser.add_argument("--window_size", default=5, type=int)
    parser.add_argument('--seed', type=int, default=100, help='random seed')

    parser.add_argument('--lr', type=float, default=1e-6, metavar='LR', help='learning rate')  #####
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=4, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=5, metavar='E', help='number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--emb_dim', type=int, default=768, help='Feature size.')
    parser.add_argument('--num_of_heads', type=int, default=8, help='num of heads.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--max_text_length', type=int, default=128, help='max length of Input utterance.')
    ### ### ###
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = []
    dev_dataset = []
    test_dataset = []
    random.seed(42)
    torch.manual_seed(42)
    train_dataset1, train_labels1, dev_dataset1, dev_labels1, test_dataset1 = get_rhe_datasets_stac()
    train_dataset2, train_labels2, dev_dataset2, dev_labels2, test_dataset2 = get_rhe_datasets_molweni()
    _, _, test_files3, train_dataset3, dev_dataset3, test_dataset3 = get_top_datasets1()
    train_dataset4, _, dev_dataset4, _, test_dataset4, test_labels4 = get_top_dataset2()
    print(len(train_dataset1),  len(dev_dataset1), len(train_labels1), len(train_dataset2), len(dev_dataset2), len(train_labels2), len(train_dataset3), len(dev_dataset3), len(train_dataset4), len(dev_dataset4))
    train_dataset = train_dataset1 + train_dataset2 + train_dataset3 + train_dataset4
    dev_dataset = dev_dataset1 + dev_dataset2 + dev_dataset3 + dev_dataset4

    random.seed(42)
    torch.manual_seed(42)
    random.shuffle(train_dataset)
    random.shuffle(dev_dataset)

    device = args.device
    n_epochs = args.epochs

    model = MixModel(args)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)

    start_time_all = time.time()
    min_val_loss = float('inf')

    with open("Log/Results/Molweni_ori_result.json", 'r') as f1:
        molweni_ori = json.load(f1)

    with open("Log/Results/stac_ori_result.json", 'r') as f1:
        stac_ori = json.load(f1)

    for e in range(n_epochs):  # 遍历每个epoch
        start_time = time.time()

        train_loss = train_or_eval_model(model, train_dataset, train_labels1, device, args, optimizer, True)
        valid_loss = train_or_eval_model(model, dev_dataset, dev_labels1, device, args)

        print('Epoch: {}, train_loss: {}, valid_loss: {}, time: {} sec'.format(e+1, train_loss, valid_loss, round(time.time() - start_time, 2)))
        PATH = f"URTmodel/urt-doc"
        model_to_save = model.module if hasattr(model, 'module') else model
        print('Saving model to ' + PATH)
        torch.save(model_to_save.state_dict(), PATH)


        all_pred_labels, all_test_labels = compute_f1(model.rhe_model)
        ori_ana, pred_ana, ori_ana2, pred_ana2, ori_ana3_4, pred_ana3_4 = analysis1(stac_ori, all_pred_labels, all_test_labels)
        print(dict(sorted(ori_ana.items(), key=lambda x: x[0])))
        print(dict(sorted(pred_ana.items(), key=lambda x: x[0])))
        print(dict(sorted(ori_ana2.items(), key=lambda x: x[0])))
        print(dict(sorted(pred_ana2.items(), key=lambda x: x[0])))
        print(ori_ana3_4)
        print(pred_ana3_4)
        all_pred_labels, all_test_labels = compute_f2(model.rhe_model)
        ori_ana, pred_ana, ori_ana2, pred_ana2, ori_ana3_4, pred_ana3_4 = analysis2(molweni_ori, all_pred_labels, all_test_labels)
        print(dict(sorted(ori_ana.items(), key=lambda x: x[0])))
        print(dict(sorted(pred_ana.items(), key=lambda x: x[0])))
        print(dict(sorted(ori_ana2.items(), key=lambda x: x[0])))
        print(dict(sorted(pred_ana2.items(), key=lambda x: x[0])))
        print(ori_ana3_4)
        print(pred_ana3_4)
        TopTestInfer1(args, model.Topmodel, test_files3)
        TopTestInfer_tiage(args, model.Topmodel, test_dataset4, test_labels4)


    print('finish training!')

    print("All Cost Time: {} seconds".format(time.time()-start_time_all))