import os
import re
import copy
import json
import torch
import pickle
import segeval
import argparse
import numpy as np
from tqdm import tqdm
from TopModel import SegModel
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
from transformers import BertTokenizer, BertForNextSentencePrediction, AutoTokenizer, set_seed
import random
from MixModel import MixModel
from dataset import get_rhe_datasets_stac, get_top_datasets_doc, get_rhe_datasets_molweni, get_top_dataset_tiage, get_top_datasets_711
from eisner import eisner
from RHE_TOP import RheInfer
DATASET = {'doc': 'doc2dial', '711': 'dialseg711'}



def depth_score_cal(scores):
    output_scores = []
    for i in range(len(scores)):
        lflag, rflag = scores[i], scores[i]
        if i == 0:
            hl = scores[i]
            for r in range(i + 1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
        elif i == len(scores) - 1:
            hr = scores[i]
            for l in range(i - 1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        else:
            for r in range(i + 1, len(scores)):
                if rflag <= scores[r]:
                    rflag = scores[r]
                else:
                    break
            for l in range(i - 1, -1, -1):
                if lflag <= scores[l]:
                    lflag = scores[l]
                else:
                    break
        depth_score = 0.5 * (lflag + rflag - 2 * scores[i])
        output_scores.append(depth_score)

    return output_scores


def infer(args, model_path, topmodel):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    if args.urt == True:
        model = topmodel
    else:
        model = SegModel()
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')), False)
    model.to(args.device)
    model.eval()
    res = {}
    path_input_docs = f'./data/{args.dataset}'
    input_files = [f for f in os.listdir(path_input_docs) if os.path.isfile(os.path.join(path_input_docs, f))]

    c = score_wd = score_pk = 0

    num = 0
    for file in tqdm(input_files):
        num += 1
        if num < 10:
            if file not in ['.DS_Store']:
                text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                        []], [
                                                                                                           [], []], [[], []]
                depth_scores = []
                lengths1 = []
                lengths2 = []
                seg_r_labels, seg_r = [], []
                tmp = 0
                for line in open(path_input_docs + '/' + file):
                    if '================' not in line.strip():
                        text.append(line.strip())
                        seg_r_labels.append(0)
                        tmp += 1
                    else:
                        seg_r_labels[-1] = 1
                        seg_r.append(tmp)
                        tmp = 0
                seg_r.append(tmp)

                for i in range(len(text) - 1):
                    context, cur = [], []
                    l, r = i, i + 1
                    for win in range(args.window_size):
                        if l > -1:
                            context.append(text[l][:128])
                            l -= 1
                        if r < len(text):
                            cur.append(text[r][:128])
                            r += 1
                    context.reverse()

                    topic_con = tokenizer(context, truncation=True, padding=True, max_length=256, return_tensors='pt')
                    topic_cur = tokenizer(cur, truncation=True, padding=True, max_length=256, return_tensors='pt')

                    topic_input[0].extend(topic_con['input_ids'])
                    topic_input[1].extend(topic_cur['input_ids'])
                    topic_att_mask[0].extend(topic_con['attention_mask'])
                    topic_att_mask[1].extend(topic_cur['attention_mask'])
                    topic_num[0].append(len(context))
                    topic_num[1].append(len(cur))

                    sent1 = ''
                    for sen in context:
                        sent1 += sen + '[SEP]'

                    sent2 = text[i + 1]

                    encoded_sent1 = tokenizer.encode(sent1, add_special_tokens=True, max_length=256, return_tensors='pt')
                    encoded_sent2 = tokenizer.encode(sent2, add_special_tokens=True, max_length=256, return_tensors='pt')
                    lengths1.append(len(encoded_sent1[0].tolist()[:-1]))
                    lengths2.append(len(encoded_sent2[0].tolist()[1:]))
                    encoded_pair = encoded_sent1[0].tolist()[:-1] + encoded_sent2[0].tolist()[1:]
                    type_id = [0] * len(encoded_sent1[0].tolist()[:-1]) + [1] * len(encoded_sent2[0].tolist()[1:])
                    type_ids.append(torch.Tensor(type_id))
                    id_inputs.append(torch.Tensor(encoded_pair))

                MAX_LEN = 512
                id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post",
                                          padding="post")
                type_ids = pad_sequences(type_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")
                for sent in id_inputs:
                    att_mask = [int(token_id > 0) for token_id in sent]
                    coheren_att_masks.append(att_mask)

                try:
                    topic_input = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_input]
                    topic_mask = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_att_mask]
                except:
                    print(file)
                    continue

                with torch.no_grad():
                    coheren_inputs = torch.tensor(id_inputs).to(args.device)
                    coheren_masks = torch.tensor(coheren_att_masks).to(args.device)
                    coheren_type_ids = torch.tensor(type_ids).to(args.device)
                    lengths1 = torch.tensor(lengths1).to(args.device)
                    lengths2 = torch.tensor(lengths2).to(args.device)
                    _, _, _, _, scores = model.infer(lengths1, lengths2, coheren_inputs, coheren_masks, coheren_type_ids, topic_input,
                                         topic_mask, topic_num)

                depth_scores = depth_score_cal(scores)
                depth_scores = [depth_score.detach().cpu().numpy() for depth_score in depth_scores]
                # print("depth_scores: ", len(depth_scores), depth_scores)
                boundary_indice = np.argsort(np.array(depth_scores))[-args.pick_num:]
                seg_p_labels = [0] * (len(depth_scores) + 1)
                for i in boundary_indice:
                    seg_p_labels[i] = 1

                tmp = 0;
                seg_p = []
                for fake in seg_p_labels:
                    if fake == 1:
                        tmp += 1
                        seg_p.append(tmp)
                        tmp = 0
                    else:
                        tmp += 1
                seg_p.append(tmp)

                print(seg_p, seg_r)
                score_wd += segeval.window_diff(seg_p, seg_r)
                score_pk += segeval.pk(seg_p, seg_r)

                c += 1

    print('pk: ', score_pk / c)
    print('wd: ', score_wd / c)
    res = str(round(float(score_pk / c), 4)) + "\t" + str(round(float(score_wd / c), 4))
    print(f'Saving result to {args.root}/metric/{args.model}/{args.save_name}.json')
    json.dump(res, open(f'{args.root}/metric/{args.model}/{args.save_name}.json', 'w'))


def TopTestInfer_711(args, topmodel, input_files):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model = topmodel
    model.to(args.device)
    model.eval()
    res = {}
    path_input_docs = './data/dialseg711'
    if input_files == None:
        input_files = [f for f in os.listdir(path_input_docs) if os.path.isfile(os.path.join(path_input_docs, f))]

    c = score_wd = score_pk = 0

    num = 0
    for file in tqdm(input_files):
        num += 1
        if num < 10000:
            if file not in ['.DS_Store']:
                text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                        []], [
                                                                                                           [], []], [[], []]
                depth_scores = []
                lengths1 = []
                lengths2 = []
                seg_r_labels, seg_r = [], []
                tmp = 0
                for line in open(path_input_docs + '/' + file):
                    if '================' not in line.strip():
                        text.append(line.strip())
                        seg_r_labels.append(0)
                        tmp += 1
                    else:
                        seg_r_labels[-1] = 1
                        seg_r.append(tmp)
                        tmp = 0
                seg_r.append(tmp)

                for i in range(len(text) - 1):
                    context, cur = [], []
                    l, r = i, i + 1
                    for win in range(args.window_size):
                        if l > -1:
                            context.append(text[l][:128])
                            l -= 1
                        if r < len(text):
                            cur.append(text[r][:128])
                            r += 1
                    context.reverse()

                    topic_con = tokenizer(context, truncation=True, padding=True, max_length=256, return_tensors='pt')
                    topic_cur = tokenizer(cur, truncation=True, padding=True, max_length=256, return_tensors='pt')

                    topic_input[0].extend(topic_con['input_ids'])
                    topic_input[1].extend(topic_cur['input_ids'])
                    topic_att_mask[0].extend(topic_con['attention_mask'])
                    topic_att_mask[1].extend(topic_cur['attention_mask'])
                    topic_num[0].append(len(context))
                    topic_num[1].append(len(cur))

                    sent1 = ''
                    for sen in context:
                        sent1 += sen + '[SEP]'

                    sent2 = text[i + 1]

                    encoded_sent1 = tokenizer.encode(sent1, add_special_tokens=True, max_length=256, return_tensors='pt')
                    encoded_sent2 = tokenizer.encode(sent2, add_special_tokens=True, max_length=256, return_tensors='pt')
                    lengths1.append(len(encoded_sent1[0].tolist()[:-1]))
                    lengths2.append(len(encoded_sent2[0].tolist()[1:]))
                    encoded_pair = encoded_sent1[0].tolist()[:-1] + encoded_sent2[0].tolist()[1:]
                    type_id = [0] * len(encoded_sent1[0].tolist()[:-1]) + [1] * len(encoded_sent2[0].tolist()[1:])
                    type_ids.append(torch.Tensor(type_id))
                    id_inputs.append(torch.Tensor(encoded_pair))

                MAX_LEN = args.max_text_length
                id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post",
                                          padding="post")
                type_ids = pad_sequences(type_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")
                for sent in id_inputs:
                    att_mask = [int(token_id > 0) for token_id in sent]
                    coheren_att_masks.append(att_mask)

                try:
                    topic_input = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_input]
                    topic_mask = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_att_mask]
                except:
                    print(file)
                    continue

                with torch.no_grad():
                    coheren_inputs = torch.tensor(id_inputs).to(args.device)
                    coheren_masks = torch.tensor(coheren_att_masks).to(args.device)
                    coheren_type_ids = torch.tensor(type_ids).to(args.device)
                    lengths1 = torch.tensor(lengths1).to(args.device)
                    lengths2 = torch.tensor(lengths2).to(args.device)
                    _, _, _, _, scores = model.infer(lengths1, lengths2, coheren_inputs, coheren_masks, coheren_type_ids,
                                         topic_input, topic_mask, topic_num)
                # print("scores: ", scores)
                depth_scores = depth_score_cal(scores)
                # print(scores)
                depth_scores = [depth_score.detach().cpu().numpy() for depth_score in depth_scores]
                # print("depth_scores: ", len(depth_scores), depth_scores)
                boundary_indice = np.argsort(np.array(depth_scores))[-args.pick_num:]
                seg_p_labels = [0] * (len(depth_scores) + 1)
                for i in boundary_indice:
                    seg_p_labels[i] = 1
                tmp = 0
                seg_p = []
                for fake in seg_p_labels:
                    if fake == 1:
                        tmp += 1
                        seg_p.append(tmp)
                        tmp = 0
                    else:
                        tmp += 1
                seg_p.append(tmp)

                print(seg_p, seg_r)
                score_wd += segeval.window_diff(seg_p, seg_r)
                score_pk += segeval.pk(seg_p, seg_r)

                c += 1

                # rhe_matrix, e_matrix = RheInfer(args.device, tokenizer, rhe_model, text)
                # result = eisner(rhe_matrix)
                # labels = []
                # for element in result:
                #     labels.append(list(element))
                # print(labels)

    print('pk: ', score_pk / c)
    print('wd: ', score_wd / c)
    res = str(round(float(score_pk / c), 4)) + "\t" + str(round(float(score_wd / c), 4))
    # print(f'Saving result to {args.root}/metric/{args.model}/{args.save_name}.json')
    # json.dump(res, open(f'{args.root}/metric/{args.model}/{args.save_name}.json', 'w'))


def TopTestInfer_711_2(args, topmodel, input_files):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model = topmodel
    model.to(args.device)
    model.eval()
    res = {}
    path_input_docs = './data/dialseg711'
    if input_files == None:
        input_files = [f for f in os.listdir(path_input_docs) if os.path.isfile(os.path.join(path_input_docs, f))]

    c = score_wd = score_pk = 0

    num = 0
    for file in tqdm(input_files):
        num += 1
        if num < 10000:
            if file not in ['.DS_Store']:
                text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                        []], [
                                                                                                           [], []], [[], []]
                depth_scores = []
                lengths1 = []
                lengths2 = []
                seg_r_labels, seg_r = [], []
                tmp = 0
                for line in open(path_input_docs + '/' + file):
                    if '================' not in line.strip():
                        text.append(line.strip())
                        seg_r_labels.append(0)
                        tmp += 1
                    else:
                        seg_r_labels[-1] = 1
                        seg_r.append(tmp)
                        tmp = 0
                seg_r.append(tmp)

                _, top_matrix, _ = topmodel(text, None, True)
                scores = torch.zeros(len(text) - 1)
                for i in range(len(text)-1):
                    scores[i] = top_matrix[i, i+1]
                # print("scores: ", scores)
                depth_scores = depth_score_cal(scores)
                # print(scores)
                depth_scores = [depth_score.detach().cpu().numpy() for depth_score in depth_scores]
                # print("depth_scores: ", len(depth_scores), depth_scores)
                boundary_indice = np.argsort(np.array(depth_scores))[-args.pick_num:]
                seg_p_labels = [0] * (len(depth_scores) + 1)
                for i in boundary_indice:
                    seg_p_labels[i] = 1
                tmp = 0
                seg_p = []
                for fake in seg_p_labels:
                    if fake == 1:
                        tmp += 1
                        seg_p.append(tmp)
                        tmp = 0
                    else:
                        tmp += 1
                seg_p.append(tmp)

                print(seg_p, seg_r)
                score_wd += segeval.window_diff(seg_p, seg_r)
                score_pk += segeval.pk(seg_p, seg_r)

                c += 1

                # rhe_matrix, e_matrix = RheInfer(args.device, tokenizer, rhe_model, text)
                # result = eisner(rhe_matrix)
                # labels = []
                # for element in result:
                #     labels.append(list(element))
                # print(labels)

    print('pk: ', score_pk / c)
    print('wd: ', score_wd / c)
    res = str(round(float(score_pk / c), 4)) + "\t" + str(round(float(score_wd / c), 4))
    # print(f'Saving result to {args.root}/metric/{args.model}/{args.save_name}.json')
    # json.dump(res, open(f'{args.root}/metric/{args.model}/{args.save_name}.json', 'w'))


def TopTestInfer_tiage(args, topmodel, test_dataset, test_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model = topmodel
    model.to(args.device)
    model.eval()
    res = {}
    c = score_wd = score_pk = 0

    num = 0
    for data in tqdm(test_dataset):
        num += 1
        if num < 10000:
            labels = test_labels[num-1]
            # if file not in ['.DS_Store']:
            id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [[],
                                                                                                                    []], [
                                                                                                       [], []], [[], []]
            depth_scores = []
            lengths1 = []
            lengths2 = []
            seg_r_labels = []
            tmp = 0
            #     for line in open(path_input_docs + '/' + file):
            #         if '================' not in line.strip():
            #             text.append(line.strip())
            #             seg_r_labels.append(0)
            #             tmp += 1
            #         else:
            #             seg_r_labels[-1] = 1
            #             seg_r.append(tmp)
            #             tmp = 0
            #     seg_r.append(tmp)

            for i in range(len(data) - 1):
                context, cur = [], []
                l, r = i, i + 1
                for win in range(args.window_size):
                    if l > -1:
                        context.append(data[l][:128])
                        l -= 1
                    if r < len(data):
                        cur.append(data[r][:128])
                        r += 1
                context.reverse()

                topic_con = tokenizer(context, truncation=True, padding=True, max_length=256, return_tensors='pt')
                topic_cur = tokenizer(cur, truncation=True, padding=True, max_length=256, return_tensors='pt')

                topic_input[0].extend(topic_con['input_ids'])
                topic_input[1].extend(topic_cur['input_ids'])
                topic_att_mask[0].extend(topic_con['attention_mask'])
                topic_att_mask[1].extend(topic_cur['attention_mask'])
                topic_num[0].append(len(context))
                topic_num[1].append(len(cur))

                sent1 = ''
                for sen in context:
                    sent1 += sen + '[SEP]'

                sent2 = data[i + 1]

                encoded_sent1 = tokenizer.encode(sent1, add_special_tokens=True, max_length=256, return_tensors='pt')
                encoded_sent2 = tokenizer.encode(sent2, add_special_tokens=True, max_length=256, return_tensors='pt')
                lengths1.append(len(encoded_sent1[0].tolist()[:-1]))
                lengths2.append(len(encoded_sent2[0].tolist()[1:]))
                encoded_pair = encoded_sent1[0].tolist()[:-1] + encoded_sent2[0].tolist()[1:]
                type_id = [0] * len(encoded_sent1[0].tolist()[:-1]) + [1] * len(encoded_sent2[0].tolist()[1:])
                type_ids.append(torch.Tensor(type_id))
                id_inputs.append(torch.Tensor(encoded_pair))

            MAX_LEN = args.max_text_length
            id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post",
                                      padding="post")
            type_ids = pad_sequences(type_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")
            for sent in id_inputs:
                att_mask = [int(token_id > 0) for token_id in sent]
                coheren_att_masks.append(att_mask)

            try:
                topic_input = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_input]
                topic_mask = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_att_mask]
            except:
                print(data)
                continue

            with torch.no_grad():
                coheren_inputs = torch.tensor(id_inputs).to(args.device)
                coheren_masks = torch.tensor(coheren_att_masks).to(args.device)
                coheren_type_ids = torch.tensor(type_ids).to(args.device)
                lengths1 = torch.tensor(lengths1).to(args.device)
                lengths2 = torch.tensor(lengths2).to(args.device)
                _, _, _, _, scores = model.infer(lengths1, lengths2, coheren_inputs, coheren_masks, coheren_type_ids, topic_input,
                                     topic_mask, topic_num)

            depth_scores = depth_score_cal(scores)
            # print(scores)
            depth_scores = [depth_score.detach().cpu().numpy() for depth_score in depth_scores]
            # print("depth_scores: ", len(depth_scores), depth_scores)
            boundary_indice = np.argsort(np.array(depth_scores))[-args.pick_num:]
            seg_p_labels = [0] * (len(depth_scores) + 1)
            for i in boundary_indice:
                seg_p_labels[i] = 1
            tmp = 0
            seg_p = []
            for fake in seg_p_labels:
                if fake == 1:
                    tmp += 1
                    seg_p.append(tmp)
                    tmp = 0
                else:
                    tmp += 1
            seg_p.append(tmp)

            tmp = 0
            seg_r = []
            for fake in labels:
                if fake == 1:
                    tmp += 1
                    seg_r.append(tmp)
                    tmp = 0
                else:
                    tmp += 1
            seg_r.append(tmp)

            # print(len(data), len(labels), seg_p, seg_r)
            score_wd += segeval.window_diff(seg_p, seg_r)
            score_pk += segeval.pk(seg_p, seg_r)

            c += 1

    print('pk: ', score_pk / c)
    print('wd: ', score_wd / c)
    res = str(round(float(score_pk / c), 4)) + "\t" + str(round(float(score_wd / c), 4))


def TopTestInfer_tiage_2(args, topmodel, test_dataset, test_labels):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model = topmodel
    model.to(args.device)
    model.eval()
    res = {}
    c = score_wd = score_pk = 0

    num = 0
    for data in tqdm(test_dataset):
        num += 1
        if num < 10000:
            labels = test_labels[num-1]
            data = data
            # if file not in ['.DS_Store']:
            id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [[],
                                                                                                                    []], [
                                                                                                       [], []], [[], []]
            depth_scores = []
            lengths1 = []
            lengths2 = []
            seg_r_labels = []
            tmp = 0
            #     for line in open(path_input_docs + '/' + file):
            #         if '================' not in line.strip():
            #             text.append(line.strip())
            #             seg_r_labels.append(0)
            #             tmp += 1
            #         else:
            #             seg_r_labels[-1] = 1
            #             seg_r.append(tmp)
            #             tmp = 0
            #     seg_r.append(tmp)

            _, top_matrix, _ = topmodel(data, None, True)
            scores = torch.zeros(len(data) - 1)
            for i in range(len(data) - 1):
                scores[i] = top_matrix[i, i + 1]

            depth_scores = depth_score_cal(scores)
            # print(scores)
            depth_scores = [depth_score.detach().cpu().numpy() for depth_score in depth_scores]
            # print("depth_scores: ", len(depth_scores), depth_scores)
            boundary_indice = np.argsort(np.array(depth_scores))[-args.pick_num:]
            seg_p_labels = [0] * (len(depth_scores) + 1)
            for i in boundary_indice:
                seg_p_labels[i] = 1
            tmp = 0
            seg_p = []
            for fake in seg_p_labels:
                if fake == 1:
                    tmp += 1
                    seg_p.append(tmp)
                    tmp = 0
                else:
                    tmp += 1
            seg_p.append(tmp)

            tmp = 0
            seg_r = []
            for fake in labels:
                if fake == 1:
                    tmp += 1
                    seg_r.append(tmp)
                    tmp = 0
                else:
                    tmp += 1
            seg_r.append(tmp)

            # print(len(data), len(labels), seg_p, seg_r)
            score_wd += segeval.window_diff(seg_p, seg_r)
            score_pk += segeval.pk(seg_p, seg_r)

            c += 1

    print('pk: ', score_pk / c)
    print('wd: ', score_wd / c)
    res = str(round(float(score_pk / c), 4)) + "\t" + str(round(float(score_wd / c), 4))


def compute_boundary(args, matrix):
    scores = []
    for i in range(matrix.size(0)-1):
        scores.append(matrix[i, i+1])
    output_scores = depth_score_cal(scores)
    print("Scores: ", output_scores)
    output_scores = [output_score.detach().cpu().numpy() for output_score in output_scores]
    boundary_indice = np.argsort(np.array(output_scores))[-args.pick_num:]
    seg_p_labels = [0] * (len(scores) + 1)
    for i in boundary_indice:
        seg_p_labels[i] = 1

    tmp = 0
    seg_p = []
    for fake in seg_p_labels:
        if fake == 1:
            tmp += 1
            seg_p.append(tmp)
            tmp = 0
        else:
            tmp += 1
    seg_p.append(tmp)

    return seg_p


def TopTestInfer2(args, model, input_files):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model.to(args.device)
    model.eval()
    res = {}
    path_input_docs = './data/doc2dial'
    if input_files == None:
        input_files = [f for f in os.listdir(path_input_docs) if os.path.isfile(os.path.join(path_input_docs, f))]

    c = score_wd1 = score_pk1 = score_wd2 = score_pk2 = score_wd3 = score_pk3 = score_wd4 = score_pk4 = 0

    num = 0
    for file in tqdm(input_files):
        num += 1
        if num < 10000:
            if file not in ['.DS_Store']:
                text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                        []], [
                                                                                                           [], []], [[], []]
                depth_scores = []
                lengths1 = []
                lengths2 = []
                seg_r_labels, seg_r = [], []
                tmp = 0
                num1 = 0
                for line in open(path_input_docs + '/' + file):
                    num1 += 1
                    if num1 < 10:
                        if '================' not in line.strip():
                            text.append(line.strip())
                            seg_r_labels.append(0)
                            tmp += 1
                        else:
                            seg_r_labels[-1] = 1
                            seg_r.append(tmp)
                            tmp = 0
                seg_r.append(tmp)

                loss, top_rhe_matrix_norm, top_rhe_matrix1, top_rhe_matrix2, top_matrix = model(text)
                seg_p1 = compute_boundary(args, top_matrix)
                seg_p2 = compute_boundary(args, top_rhe_matrix1)
                seg_p3 = compute_boundary(args, top_rhe_matrix2)
                seg_p4 = compute_boundary(args, top_rhe_matrix_norm)



                print(seg_p1, seg_r)
                print(seg_p2, seg_r)
                print(seg_p3, seg_r)
                print(seg_p4, seg_r)
                score_wd1 += segeval.window_diff(seg_p1, seg_r)
                score_pk1 += segeval.pk(seg_p1, seg_r)
                score_wd2 += segeval.window_diff(seg_p2, seg_r)
                score_pk2 += segeval.pk(seg_p2, seg_r)
                score_wd3 += segeval.window_diff(seg_p3, seg_r)
                score_pk3 += segeval.pk(seg_p3, seg_r)
                score_wd4 += segeval.window_diff(seg_p4, seg_r)
                score_pk4 += segeval.pk(seg_p4, seg_r)

                c += 1

    print('pk: ', score_pk1 / c, score_pk2 / c, score_pk3 / c, score_pk4 / c)
    print('wd: ', score_wd1 / c, score_wd2 / c, score_wd3 / c, score_wd4 / c)
    # print(f'Saving result to {args.root}/metric/{args.model}/{args.save_name}.json')
    # json.dump(res, open(f'{args.root}/metric/{args.model}/{args.save_name}.json', 'w'))


def TopTestInfer_doc(args, topmodel, input_files):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model = topmodel
    model.to(args.device)
    model.eval()
    res = {}
    path_input_docs = './data/doc2dial'
    if input_files == None:
        input_files = [f for f in os.listdir(path_input_docs) if os.path.isfile(os.path.join(path_input_docs, f))]

    c = score_wd = score_pk = 0

    num = 0
    for file in tqdm(input_files):
        num += 1
        if num < 10000:
            if file not in ['.DS_Store']:
                text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                        []], [
                                                                                                           [], []], [[], []]
                depth_scores = []
                lengths1 = []
                lengths2 = []
                seg_r_labels, seg_r = [], []
                tmp = 0
                for line in open(path_input_docs + '/' + file):
                    if '================' not in line.strip():
                        text.append(line.strip())
                        seg_r_labels.append(0)
                        tmp += 1
                    else:
                        seg_r_labels[-1] = 1
                        seg_r.append(tmp)
                        tmp = 0
                seg_r.append(tmp)

                for i in range(len(text) - 1):
                    context, cur = [], []
                    l, r = i, i + 1
                    for win in range(args.window_size):
                        if l > -1:
                            context.append(text[l][:128])
                            l -= 1
                        if r < len(text):
                            cur.append(text[r][:128])
                            r += 1
                    context.reverse()

                    topic_con = tokenizer(context, truncation=True, padding=True, max_length=256, return_tensors='pt')
                    topic_cur = tokenizer(cur, truncation=True, padding=True, max_length=256, return_tensors='pt')

                    topic_input[0].extend(topic_con['input_ids'])
                    topic_input[1].extend(topic_cur['input_ids'])
                    topic_att_mask[0].extend(topic_con['attention_mask'])
                    topic_att_mask[1].extend(topic_cur['attention_mask'])
                    topic_num[0].append(len(context))
                    topic_num[1].append(len(cur))

                    sent1 = ''
                    for sen in context:
                        sent1 += sen + '[SEP]'

                    sent2 = text[i + 1]

                    encoded_sent1 = tokenizer.encode(sent1, add_special_tokens=True, max_length=256, return_tensors='pt')
                    encoded_sent2 = tokenizer.encode(sent2, add_special_tokens=True, max_length=256, return_tensors='pt')
                    lengths1.append(len(encoded_sent1[0].tolist()[:-1]))
                    lengths2.append(len(encoded_sent2[0].tolist()[1:]))
                    encoded_pair = encoded_sent1[0].tolist()[:-1] + encoded_sent2[0].tolist()[1:]
                    type_id = [0] * len(encoded_sent1[0].tolist()[:-1]) + [1] * len(encoded_sent2[0].tolist()[1:])
                    type_ids.append(torch.Tensor(type_id))
                    id_inputs.append(torch.Tensor(encoded_pair))

                MAX_LEN = args.max_text_length
                id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post",
                                          padding="post")
                type_ids = pad_sequences(type_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")
                for sent in id_inputs:
                    att_mask = [int(token_id > 0) for token_id in sent]
                    coheren_att_masks.append(att_mask)

                try:
                    topic_input = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_input]
                    topic_mask = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_att_mask]
                except:
                    print(file)
                    continue

                with torch.no_grad():
                    coheren_inputs = torch.tensor(id_inputs).to(args.device)
                    coheren_masks = torch.tensor(coheren_att_masks).to(args.device)
                    coheren_type_ids = torch.tensor(type_ids).to(args.device)
                    lengths1 = torch.tensor(lengths1).to(args.device)
                    lengths2 = torch.tensor(lengths2).to(args.device)
                    _, _, _, _, scores = model.infer(lengths1, lengths2, coheren_inputs, coheren_masks, coheren_type_ids,
                                         topic_input, topic_mask, topic_num)

                depth_scores = depth_score_cal(scores)
                # print(scores)
                depth_scores = [depth_score.detach().cpu().numpy() for depth_score in depth_scores]
                # print("depth_scores: ", len(depth_scores), depth_scores)
                boundary_indice = np.argsort(np.array(depth_scores))[-args.pick_num:]
                seg_p_labels = [0] * (len(depth_scores) + 1)
                for i in boundary_indice:
                    seg_p_labels[i] = 1
                tmp = 0
                seg_p = []
                for fake in seg_p_labels:
                    if fake == 1:
                        tmp += 1
                        seg_p.append(tmp)
                        tmp = 0
                    else:
                        tmp += 1
                seg_p.append(tmp)

                print(seg_p, seg_r)
                score_wd += segeval.window_diff(seg_p, seg_r)
                score_pk += segeval.pk(seg_p, seg_r)

                c += 1

    print('pk: ', score_pk / c)
    print('wd: ', score_wd / c)
    res = str(round(float(score_pk / c), 4)) + "\t" + str(round(float(score_wd / c), 4))
    # print(f'Saving result to {args.root}/metric/{args.model}/{args.save_name}.json')
    # json.dump(res, open(f'{args.root}/metric/{args.model}/{args.save_name}.json', 'w'))


def TopTestInfer_doc_2(args, topmodel, input_files):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    model = topmodel
    model.to(args.device)
    model.eval()
    res = {}
    path_input_docs = './data/doc2dial'
    if input_files == None:
        input_files = [f for f in os.listdir(path_input_docs) if os.path.isfile(os.path.join(path_input_docs, f))]

    c = score_wd = score_pk = 0

    num = 0
    for file in tqdm(input_files):
        num += 1
        if num < 10000:
            if file not in ['.DS_Store']:
                text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                        []], [
                                                                                                           [], []], [[], []]
                depth_scores = []
                lengths1 = []
                lengths2 = []
                seg_r_labels, seg_r = [], []
                tmp = 0
                for line in open(path_input_docs + '/' + file):
                    if '================' not in line.strip():
                        text.append(line.strip())
                        seg_r_labels.append(0)
                        tmp += 1
                    else:
                        seg_r_labels[-1] = 1
                        seg_r.append(tmp)
                        tmp = 0
                seg_r.append(tmp)

                _, top_matrix, _ = topmodel(text, None, True)
                scores = torch.zeros(len(text) - 1)
                for i in range(len(text) - 1):
                    scores[i] = top_matrix[i, i + 1]

                depth_scores = depth_score_cal(scores)
                # print(scores)
                depth_scores = [depth_score.detach().cpu().numpy() for depth_score in depth_scores]
                # print("depth_scores: ", len(depth_scores), depth_scores)
                boundary_indice = np.argsort(np.array(depth_scores))[-args.pick_num:]
                seg_p_labels = [0] * (len(depth_scores) + 1)
                for i in boundary_indice:
                    seg_p_labels[i] = 1
                tmp = 0
                seg_p = []
                for fake in seg_p_labels:
                    if fake == 1:
                        tmp += 1
                        seg_p.append(tmp)
                        tmp = 0
                    else:
                        tmp += 1
                seg_p.append(tmp)

                print(seg_p, seg_r)
                score_wd += segeval.window_diff(seg_p, seg_r)
                score_pk += segeval.pk(seg_p, seg_r)

                c += 1

    print('pk: ', score_pk / c)
    print('wd: ', score_wd / c)
    res = str(round(float(score_pk / c), 4)) + "\t" + str(round(float(score_wd / c), 4))
    # print(f'Saving result to {args.root}/metric/{args.model}/{args.save_name}.json')
    # json.dump(res, open(f'{args.root}/metric/{args.model}/{args.save_name}.json', 'w'))


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
    parser.add_argument('--lr', type=float, default=3e-5, metavar='LR', help='learning rate')  #####
    parser.add_argument('--dropout', type=float, default=0.4, metavar='dropout', help='dropout rate')
    parser.add_argument('--batch_size', type=int, default=4, metavar='BS', help='batch size')
    parser.add_argument('--epochs', type=int, default=5, metavar='E', help='number of epochs')
    parser.add_argument('--hidden_dim', type=int, default=300)
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of gnn layers.')
    parser.add_argument('--emb_dim', type=int, default=768, help='Feature size.')
    parser.add_argument('--num_of_heads', type=int, default=8, help='num of heads.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--max_text_length', type=int, default=128, help='max length of Input utterance.')
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    train_dataset = []
    dev_dataset = []
    test_dataset = []
    # random.seed(42)
    # torch.manual_seed(42)
    set_seed(42)
    _, _, test_files, _, _, _ = get_top_datasets_711()
    model = MixModel(args)
    model.load_state_dict(torch.load(f'/mntnfs/lee_data1/xujiahui/Ablation/STAC-1e-6-12', map_location='cuda'), False)
    # model.load_state_dict(torch.load(f'model/model_doc', map_location='cuda'), False)
    Topmodel = model.Topmodel
    rhe_model = model.rhe_model
    # Topmodel.load_state_dict(torch.load('model/model_doc', map_location='cuda'), False)
    for i in range(1):
        set_seed(42)
        TopTestInfer_711(args, Topmodel, rhe_model, test_files)
