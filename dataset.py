import os
from tqdm import tqdm
import random
import json
from eda import eda
from transformers import set_seed


def check_for_letters(input_string):
    for char in input_string:
        if char.isalpha():
            return True
    return False


def collate_fn(batch):
    return batch


def get_rhe_datasets_stac():
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    train_labels = []
    dev_labels = []
    with open('data/STAC/train.json', 'r') as f1:
        data = json.load(f1)
        num = 0
        for i in data:
            num += 1
            texts = []
            texts2 = []
            texts3 = []
            for j in i['edus']:
                texts.append(j['text'])
                if check_for_letters(j['text']) and len(j['text']) > 5:
                    texts2.append(eda(sentence=j['text'])[0])
                    texts3.append(eda(sentence=j['text'])[1])
                else:
                    texts2.append(j['text'])
                    texts3.append(j['text'])

            if len(texts) > 2 and len(texts) < 18:
                train_dataset.append(texts)
                test_labels = []
                for r in i['relations']:
                    test_labels.append([r['x'], r['y']])
                train_labels.append(test_labels)
                # train_dataset.append(texts2)
                # train_dataset.append(texts3)

    # random.shuffle(train_dataset)

    with open('data/STAC/dev.json', 'r') as f2:
        data = json.load(f2)
        for i in data:
            texts = []
            texts2 = []
            texts3 = []
            for j in i['edus']:
                texts.append(j['text'])
                if check_for_letters(j['text']) and len(j['text']) > 5:
                    texts2.append(eda(sentence=j['text'])[0])
                    texts3.append(eda(sentence=j['text'])[1])
                else:
                    texts2.append(j['text'])
                    texts3.append(j['text'])

            if len(texts) > 2 and len(texts) < 18:
                dev_dataset.append(texts)
                test_labels = []
                for r in i['relations']:
                    test_labels.append([r['x'], r['y']])
                dev_labels.append(test_labels)
                # dev_dataset.append(texts2)
                # dev_dataset.append(texts3)

    with open('data/STAC/test.json', 'r') as f3:
        data = json.load(f3)
        for i in data:
            texts = []
            for j in i['edus']:
                texts.append(j['text'])
            if len(texts) > 1 and len(texts) < 100:
                test_dataset.append(texts)

    return train_dataset, train_labels, dev_dataset, dev_labels, test_dataset


def get_rhe_datasets_molweni():
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    train_labels = []
    dev_labels = []
    with open('data/Molweni/train.json', 'r') as f1:
        data = json.load(f1)
        num = 0
        for i in data:
            num += 1
            texts = []
            texts2 = []
            texts3 = []
            for j in i['edus']:
                texts.append(j['text'])
                # if check_for_letters(j['text']) and len(j['text']) > 5:
                #     texts2.append(eda(sentence=j['text'])[0])
                #     texts3.append(eda(sentence=j['text'])[1])
                # else:
                #     texts2.append(j['text'])
                #     texts3.append(j['text'])

            if len(texts) > 2 and len(texts) < 10:
                train_dataset.append(texts)
                test_labels = []
                for r in i['relations']:
                    test_labels.append([r['x'], r['y']])
                train_labels.append(test_labels)
                # train_dataset.append(texts2)
                # train_dataset.append(texts3)

    # random.shuffle(train_dataset)

    with open('data/Molweni/dev.json', 'r') as f2:
        data = json.load(f2)
        for i in data:
            texts = []
            for j in i['edus']:
                texts.append(j['text'])
            if len(texts) > 2 and len(texts) < 10:
                dev_dataset.append(texts)
                test_labels = []
                for r in i['relations']:
                    test_labels.append([r['x'], r['y']])
                dev_labels.append(test_labels)
    with open('data/Molweni/test.json', 'r') as f3:
        data = json.load(f3)
        for i in data:
            texts = []
            for j in i['edus']:
                texts.append(j['text'])
            if len(texts) > 1 and len(texts) < 100:
                test_dataset.append(texts)

    return train_dataset, train_labels, dev_dataset, dev_labels, test_dataset


def get_top_datasets_doc():
    path_input_docs = f'./data/doc2dial'
    input_files = [f for f in os.listdir(path_input_docs) if os.path.isfile(os.path.join(path_input_docs, f))]
    random.shuffle(input_files)
    # 计算各个数据集的大小
    total_size = len(input_files)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # 分割数据
    train_files = input_files[:train_size]
    dev_files = input_files[train_size:train_size + val_size]
    test_files = input_files[train_size + val_size:]

    num = 0
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for file in tqdm(train_files):
        num += 1
        if file not in ['.DS_Store']:
            text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                    []], [
                                                                                                       [], []], [[], []]

            for line in open(path_input_docs + '/' + file):
                if '================' not in line.strip():
                    if len(text) < 18:
                        text.append(line.strip())
            train_dataset.append(text)

    for file in tqdm(dev_files):
        num += 1
        if file not in ['.DS_Store']:
            text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                    []], [
                                                                                                       [], []], [[], []]

            for line in open(path_input_docs + '/' + file):
                if '================' not in line.strip():
                    if len(text) < 18:
                        text.append(line.strip())
            dev_dataset.append(text)

    for file in tqdm(test_files):
        num += 1
        if file not in ['.DS_Store']:
            text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                    []], [
                                                                                                       [], []], [[], []]

            for line in open(path_input_docs + '/' + file):
                if '================' not in line.strip():
                    text.append(line.strip())
            test_dataset.append(text)

    return train_files, dev_files, test_files, train_dataset, dev_dataset, test_dataset


def get_top_datasets_711():
    path_input_docs = f'./data/dialseg711'
    input_files = [f for f in os.listdir(path_input_docs) if os.path.isfile(os.path.join(path_input_docs, f))]
    set_seed(42)
    random.shuffle(input_files)
    # 计算各个数据集的大小
    total_size = len(input_files)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size

    # 分割数据
    train_files = input_files[:train_size]
    dev_files = input_files[train_size:train_size + val_size]
    # test_files = input_files[train_size + val_size:]
    test_files = input_files

    num = 0
    train_dataset = []
    dev_dataset = []
    test_dataset = []
    for file in tqdm(train_files):
        num += 1
        if file not in ['.DS_Store']:
            text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                    []], [
                                                                                                       [], []], [[], []]

            for line in open(path_input_docs + '/' + file):
                if '================' not in line.strip():
                    if len(text) < 8:
                        text.append(line.strip())
            train_dataset.append(text)

    for file in tqdm(dev_files):
        num += 1
        if file not in ['.DS_Store']:
            text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                    []], [
                                                                                                       [], []], [[], []]

            for line in open(path_input_docs + '/' + file):
                if '================' not in line.strip():
                    if len(text) < 7:
                        text.append(line.strip())
            dev_dataset.append(text)

    for file in tqdm(test_files):
        num += 1
        if file not in ['.DS_Store']:
            text, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[],
                                                                                                                    []], [
                                                                                                       [], []], [[], []]

            for line in open(path_input_docs + '/' + file):
                if '================' not in line.strip():
                    text.append(line.strip())
            test_dataset.append(text)

    return train_files, dev_files, test_files, train_dataset, dev_dataset, test_dataset


def get_top_dataset_tiage():
    with open('data/TIAGE/train.json', 'r') as f:
        data = json.load(f)
        train_dataset = []
        train_labels = []
        for k, v in data.items():
            dialogue = []
            dialogue2 = []
            dialogue3 = []
            dialogue4 = []
            dialogue5 = []
            dialogue6 = []
            dialogue7 = []
            tmp = []
            for u in v:
                if len(dialogue) < 18:
                    dialogue.append(u[0])
                    if check_for_letters(u[0]) and len(u[0]) > 5:
                        dialogue2.append(eda(sentence=u[0])[0])
                        dialogue3.append(eda(sentence=u[0])[1])
                        dialogue4.append(eda(sentence=u[0])[2])
                        dialogue5.append(eda(sentence=u[0])[3])
                        dialogue6.append(eda(sentence=u[0])[4])
                        dialogue7.append(eda(sentence=u[0])[5])
                    else:
                        dialogue2.append(u[0])
                        dialogue3.append(u[0])
                        dialogue4.append(u[0])
                        dialogue5.append(u[0])
                        dialogue6.append(u[0])
                        dialogue7.append(u[0])
                    tmp.append(int(u[1]))
            tmp[0] = 0
            train_dataset.append(dialogue)
            # train_dataset.append(dialogue2)
            # train_dataset.append(dialogue3)
            # train_dataset.append(dialogue4)
            # train_dataset.append(dialogue5)
            # train_dataset.append(dialogue6)
            # train_dataset.append(dialogue7)
            train_labels.append(tmp)

    with open('data/TIAGE/dev.json', 'r') as f:
        data = json.load(f)
        dev_dataset = []
        dev_labels = []
        for k, v in data.items():
            dialogue = []
            dialogue2 = []
            dialogue3 = []
            dialogue4 = []
            dialogue5 = []
            dialogue6 = []
            dialogue7 = []
            tmp = []
            for u in v:
                if len(dialogue) < 18:
                    dialogue.append(u[0])
                    if check_for_letters(u[0]) and len(u[0]) > 5:
                        dialogue2.append(eda(sentence=u[0])[0])
                        dialogue3.append(eda(sentence=u[0])[1])
                        dialogue4.append(eda(sentence=u[0])[2])
                        dialogue5.append(eda(sentence=u[0])[3])
                        dialogue6.append(eda(sentence=u[0])[4])
                        dialogue7.append(eda(sentence=u[0])[5])
                    else:
                        dialogue2.append(u[0])
                        dialogue3.append(u[0])
                        dialogue4.append(u[0])
                        dialogue5.append(u[0])
                        dialogue6.append(u[0])
                        dialogue7.append(u[0])
                    tmp.append(int(u[1]))
            tmp[0] = 0
            dev_dataset.append(dialogue)
            # dev_dataset.append(dialogue2)
            # dev_dataset.append(dialogue3)

            dev_labels.append(tmp)

    with open('data/TIAGE/test.json', 'r') as f:
        data = json.load(f)
        test_dataset = []
        test_labels = []
        for k, v in data.items():
            dialogue = []
            tmp = []
            for u in v:
                dialogue.append(u[0])
                tmp.append(int(u[1]))
            tmp[0] = 0
            test_dataset.append(dialogue)
            test_labels.append(tmp)

    return train_dataset, train_labels, dev_dataset,dev_labels, test_dataset, test_labels
