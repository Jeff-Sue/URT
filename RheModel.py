from transformers import AutoTokenizer, AutoModel, AutoConfig, BartModel, BertModel, MBartModel
import os
import copy
import torch
# import numpy as np
# import multiprocessing
# from concurrent.futures import ThreadPoolExecutor, as_completed
import json
# import pandas as pd
# import seaborn as sns
import random
import time

### Generate EDU aggregated self-attention Matrixes from BART etc.


class DiscDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, model_params):
        ### rhe_model setting
        self.model_params = ['bart-large-chinese']
        self.files = [os.path.join(file_path, x) for x in os.listdir(file_path) if x.endswith(".out.edus")]
        if 'cli' in model_params["name"]:  # fine-tuned on sentence ordering, models stored in ReBART/ repo
            model_path = os.path.join("ReBART/outputs/", model_params["name"].split('cli/')[-1])
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_params["name"],
                                                           cache_dir="./huggingface_model_files",
                                                           local_files_only=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        edu_lengths = []
        inputs = []
        slice_list = []
        with open('data/STAC/stac_situated.json', "r") as f:
            for line in f:
                line = line.strip().lower().replace("\n", "").replace("\r", "") + " "
                tokens = self.tokenizer(line)["input_ids"][
                         self.model_params["prefix_len"]:-self.model_params["postfix_len"]]
                edu_lengths.append(len(tokens))
                inputs.extend(tokens)

        if len(inputs) <= self.model_params["max_input_size"]:
            slice_list = [torch.tensor([inputs])]
        else:
            # Split input into slices of the max model input size and add start+end tokens
            for split_point in range(len(inputs) - (self.model_params["max_input_size"] - 2) + 1):
                curr_inputs = copy.deepcopy(inputs)[split_point:split_point + (self.model_params["max_input_size"] - 2)]
                curr_inputs = [self.model_params["start_token"]] + curr_inputs + [self.model_params["end_token"]]
                slice_list.append(curr_inputs)
            if len(slice_list) <= self.model_params["batch_size"]:
                slice_list = [torch.tensor(slice_list)]
            else:
                slice_list = torch.tensor(slice_list)
                slice_list = list(torch.split(slice_list, self.model_params["batch_size"]))

        # Generate huggingface required format for model inputs
        data_objects = []
        for data_slice in slice_list:
            datapoint = {}
            datapoint["input_ids"] = data_slice
            datapoint["attention_mask"] = torch.ones(data_slice.size(), dtype=torch.long)
            if self.model_params["has_token_type_ids"]:
                datapoint["token_type_ids"] = torch.zeros(data_slice.size(), dtype=torch.long)
            data_objects.append(datapoint)
        file_name = os.path.basename(self.files[index]).split(".")[0]
        return data_objects, len(inputs), edu_lengths, file_name, self.files[index]

def Mydata(data_point):
    edu_lengths = []
    inputs = []
    slice_list = []
    model_path = 'facebook/bart-large-cnn'
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    for edu in data_point:
        line = edu['text'].strip().lower().replace("\n", "").replace("\r", "") + " "
        tokens = tokenizer(line)["input_ids"][1:-1]
        edu_lengths.append(len(tokens))
        inputs.extend(tokens)
    if len(inputs) <= 1024:
        slice_list = [torch.tensor([inputs])]
    else:
        # Split input into slices of the max model input size and add start+end tokens
        for split_point in range(len(inputs) - (1024 - 2) + 1):
            curr_inputs = copy.deepcopy(inputs)[split_point:split_point + (1024 - 2)]
            curr_inputs = [0] + curr_inputs + [2]
            slice_list.append(curr_inputs)
        if len(slice_list) <= 1:
            slice_list = [torch.tensor(slice_list)]
        else:
            slice_list = torch.tensor(slice_list)
            slice_list = list(torch.split(slice_list, 1))

    data_objects = []
    for data_slice in slice_list:
        datapoint = {}
        datapoint["input_ids"] = data_slice
        datapoint["attention_mask"] = torch.ones(data_slice.size(), dtype=torch.long)
        data_objects.append(datapoint)

    return data_objects, len(inputs), edu_lengths

def squeeze_if_needed(data_slice):
    if len(data_slice.size()) == 3:
        data_slice = data_slice.squeeze(0)
    return data_slice


def execution(model, model_params, datapoint, base_save_path, layer, head, dataname, random_init, orig_parseval):
    # dataset = DiscDataset(file_path, model_params)
    # data_loader = torch.utils.data.DataLoader(dataset, shuffle=True)
    # model = None
    # start_time_all = time.time()
    # results = []
    # results1 = []
    with open(file_path, 'r') as f1:
        data = json.load(f1)
        data = data['dialogues']
        random.shuffle(data)
    data, dimensions, edu_lengths = Mydata(datapoint)
    print("dimensions and edu length: ", len(data), dimensions, edu_lengths)
    device = torch.device('cuda')
    model.to(device)
    model.eval()
    agg_self_attention = torch.zeros([dimensions, dimensions])
    agg_self_attention_divisor = torch.zeros([dimensions, dimensions])
    aggregated_outputs = []
    for data_slice in data:
        # Feed data_slice into model
        data_slice["input_ids"] = squeeze_if_needed(data_slice["input_ids"].to(device))
        data_slice["attention_mask"] = squeeze_if_needed(data_slice["attention_mask"].to(device))
        if model_params["has_token_type_ids"]:
            data_slice["token_type_ids"] = squeeze_if_needed(data_slice["token_type_ids"].to(device))
        output = model(**data_slice)
        output = output[model_params["attn_keyword"]]
        output = output[layer].detach()
        output = output[:, head, :, :]
        # Remove first and last elements, since they're not part of the input document for each batch element
        output = output[:, 1:-1, 1:-1]
        # Aggregate self-attention outputs into a single list
        aggregated_outputs.extend(list(output.cuda()))
    # Average accross layovers
    for idx, attention_slice in enumerate(aggregated_outputs):
        aggregated_attention = torch.add(
            agg_self_attention[idx:idx + attention_slice.size(0), idx:idx + attention_slice.size(1)],
            attention_slice)
        agg_self_attention[idx:idx + attention_slice.size(0),
        idx:idx + attention_slice.size(1)] = aggregated_attention
        agg_self_attention_divisor[idx:idx + attention_slice.size(0),
        idx:idx + attention_slice.size(1)] += torch.ones(aggregated_attention.size())
    agg_self_attention = agg_self_attention / agg_self_attention_divisor
    # Combine EDUs
    assert (sum(edu_lengths) == agg_self_attention.size(0)), "EDUs don't match with tokens"
    agg_self_attention = torch.nan_to_num(agg_self_attention)
    edu_attn_matrix = torch.eye(len(edu_lengths))
    first_index_agg = 0  # offset from start
    for first_idx, item1 in enumerate(edu_lengths):
        second_index_agg = 0  # offset from start
        for second_idx, item2 in enumerate(edu_lengths):
            if first_idx == second_idx:
                # Get outgoing importance
                edu_importance_pre = agg_self_attention[:first_index_agg,
                                     first_index_agg:first_index_agg + item1]
                edu_importance_post = agg_self_attention[first_index_agg + item1:,
                                      first_index_agg:first_index_agg + item1]
                edu_importance = torch.cat([edu_importance_pre, edu_importance_post])
                edu_attn_matrix[first_idx, second_idx] = torch.mean(torch.flatten(edu_importance))
                second_index_agg += item2
                continue
            else:
                # Make sure to not double count
                if float(edu_attn_matrix[first_idx, second_idx]) == 0. or float(
                        edu_attn_matrix[second_idx, first_idx]) == 0.:
                    # Incoming
                    ingoing_subtable = agg_self_attention[first_index_agg:first_index_agg + item1,
                                       second_index_agg:second_index_agg + item2]
                    ingoing_subtable = torch.mean(torch.flatten(ingoing_subtable))
                    # Outgoing
                    outgoing_subtable = agg_self_attention[second_index_agg:second_index_agg + item2,
                                        first_index_agg:first_index_agg + item1]
                    outgoing_subtable = torch.mean(torch.flatten(outgoing_subtable))
                    # Save
                    edu_attn_matrix[first_idx, second_idx] = ingoing_subtable
                    edu_attn_matrix[second_idx, first_idx] = outgoing_subtable
            second_index_agg += item2
        first_index_agg += item1

    base_data = {"layer": layer,
                 "head": head,
                 }
    mat_size = edu_attn_matrix.size(0)  # edu_attn matrix shape [n,n], n is nb of edu
    # li: modify edu_attn_matrix to reduce backdward links, only take half right upper half, the other half give 0
    half_edu_attn_matrix = torch.zeros(mat_size, mat_size)
    for i in range(mat_size):
        for j in range(mat_size):
            if i <= j:
                half_edu_attn_matrix[i, j] = edu_attn_matrix[i, j]

    # print("half_edu_attn_matrix: ", half_edu_attn_matrix.size(), half_edu_attn_matrix)
    dep_tree, weight_matrix = eisner(half_edu_attn_matrix)

    return dep_tree, weight_matrix


def single_execution(device, model, model_params, layers, heads, data, dimensions, edu_lengths):

    model.to(device)
    for data_slice in data:
        # Feed data_slice into model
        data_slice["input_ids"] = squeeze_if_needed(data_slice["input_ids"].to(device))
        data_slice["attention_mask"] = squeeze_if_needed(data_slice["attention_mask"].to(device))
        if model_params["has_token_type_ids"]:
            data_slice["token_type_ids"] = squeeze_if_needed(data_slice["token_type_ids"].to(device))
        output1 = model(**data_slice)

    matrix_list = []
    for layer in [layers-1]:
        for head in heads:
    # for layer in [layers]:
    #     for head in [heads]:
            agg_self_attention = torch.zeros([dimensions, dimensions]).to(device)
            agg_self_attention_divisor = torch.zeros([dimensions, dimensions]).to(device)
            aggregated_outputs = []
            output = output1[model_params["attn_keyword"]]
            output = output[layer]
            output = output[:, head, :, :]
            # Remove first and last elements, since they're not part of the input document for each batch element
            output = output[:, 1:-1, 1:-1]
            # Aggregate self-attention outputs into a single list
            aggregated_outputs.extend(list(output))
            # Average accross layovers
            for idx, attention_slice in enumerate(aggregated_outputs):
                aggregated_attention = torch.add(
                    agg_self_attention[idx:idx + attention_slice.size(0), idx:idx + attention_slice.size(1)],
                    attention_slice)
                agg_self_attention[idx:idx + attention_slice.size(0),
                idx:idx + attention_slice.size(1)] = aggregated_attention
                agg_self_attention_divisor[idx:idx + attention_slice.size(0),
                idx:idx + attention_slice.size(1)] = agg_self_attention_divisor[idx:idx + attention_slice.size(0),
                idx:idx + attention_slice.size(1)] + torch.ones(aggregated_attention.size()).to(device)
            agg_self_attention = agg_self_attention / agg_self_attention_divisor
            # Combine EDUs
            assert (sum(edu_lengths) == agg_self_attention.size(0)), "EDUs don't match with tokens"
            agg_self_attention = torch.nan_to_num(agg_self_attention)
            edu_attn_matrix = torch.eye(len(edu_lengths))
            first_index_agg = 0  # offset from start
            for first_idx, item1 in enumerate(edu_lengths):
                second_index_agg = 0  # offset from start
                for second_idx, item2 in enumerate(edu_lengths):
                    if first_idx == second_idx:
                        # Get outgoing importance
                        edu_importance_pre = agg_self_attention[:first_index_agg,
                                             first_index_agg:first_index_agg + item1]
                        edu_importance_post = agg_self_attention[first_index_agg + item1:,
                                              first_index_agg:first_index_agg + item1]
                        edu_importance = torch.cat([edu_importance_pre, edu_importance_post])
                        edu_attn_matrix[first_idx, second_idx] = torch.mean(torch.flatten(edu_importance))
                        second_index_agg = second_index_agg + item2
                        continue
                    else:
                        # Make sure to not double count
                        if float(edu_attn_matrix[first_idx, second_idx]) == 0. or float(
                                edu_attn_matrix[second_idx, first_idx]) == 0.:
                            # Incoming
                            ingoing_subtable = agg_self_attention[first_index_agg:first_index_agg + item1,
                                               second_index_agg:second_index_agg + item2]
                            ingoing_subtable = torch.mean(torch.flatten(ingoing_subtable))
                            # Outgoing
                            outgoing_subtable = agg_self_attention[second_index_agg:second_index_agg + item2,
                                                first_index_agg:first_index_agg + item1]
                            outgoing_subtable = torch.mean(torch.flatten(outgoing_subtable))
                            # Save
                            edu_attn_matrix[first_idx, second_idx] = ingoing_subtable
                            edu_attn_matrix[second_idx, first_idx] = outgoing_subtable
                    second_index_agg = second_index_agg + item2
                first_index_agg = first_index_agg + item1

            mat_size = edu_attn_matrix.size(0)  # edu_attn matrix shape [n,n], n is nb of edu

            half_edu_attn_matrix = torch.zeros(mat_size, mat_size)
            for i in range(mat_size):
                for j in range(mat_size):
                    if i <= j:
                        half_edu_attn_matrix[i, j] = edu_attn_matrix[i, j]
            matrix_list.append(half_edu_attn_matrix)

    # print(f"--- {(time.time() - start_time_all)} seconds to complete head/layer combo ---\n")
    return matrix_list


### Construct Dep Tree with Eisner
def eisner(G):
    num_v = G.size(0)
    weight_matrix = torch.zeros(num_v, num_v, 2, 2)
    selection_matrix = torch.zeros(num_v, num_v, 2, 2)
    for m in range(1, num_v):
        for i in range(num_v - m):
            j = i + m
            ##d=0, c=0
            max_score = 0
            max_id = -1
            for q in range(i, j):
                score = weight_matrix[i, q, 1, 1] + weight_matrix[q + 1, j, 0, 1] + G[j, i]
                if score > max_score:
                    max_score = score
                    max_id = q
            weight_matrix[i, j, 0, 0] = max_score
            selection_matrix[i, j, 0, 0] = max_id

            ##d=1, c=0
            max_score = 0
            max_id = -1
            for q in range(i, j):
                score = weight_matrix[i, q, 1, 1] + weight_matrix[q + 1, j, 0, 1] + G[i, j]
                if score > max_score:
                    max_score = score
                    max_id = q
            weight_matrix[i, j, 1, 0] = max_score
            selection_matrix[i, j, 1, 0] = max_id

            ##d=0, c=1
            max_score = 0
            max_id = -1
            for q in range(i, j + 1):
                score = weight_matrix[i, q, 0, 1] + weight_matrix[q, j, 0, 0]
                if score > max_score:
                    max_score = score
                    max_id = q
            weight_matrix[i, j, 0, 1] = max_score
            selection_matrix[i, j, 0, 1] = max_id

            ##d=1, c=1
            max_score = 0
            max_id = -1
            for q in range(i, j + 1):
                score = weight_matrix[i, q, 1, 0] + weight_matrix[q, j, 1, 1]
                if score > max_score:
                    max_score = score
                    max_id = q
            weight_matrix[i, j, 1, 1] = max_score
            selection_matrix[i, j, 1, 1] = max_id

    dep_tree = Traceback(selection_matrix, 0, num_v - 1, 1, 1)
    return dep_tree, weight_matrix


def stringify(dictionary, offset):
    string_list = []
    for key in dictionary.keys():
        for element in dictionary[key]:
            string_list.append(f"{key + offset}-{element + offset}")
    return string_list


def Traceback(selection_matrix, i, j, d, c):
    # print("i and j: ", i, j)
    if i == j:
        return {}
    q = int(selection_matrix[i, j, d, c])
    # print("Q: ", q)
    if d == 1 and c == 1:
        left_result = Traceback(selection_matrix, i, q, 1, 0)
        # print("1-1 left_result: ", left_result)
        right_result = Traceback(selection_matrix, q, j, 1, 1)
        # print("1-1 right_result: ", right_result)
        current_dep = merge_dict(left_result, right_result)
    elif d == 0 and c == 1:
        left_result = Traceback(selection_matrix, i, q, 0, 1)
        # print("0-1 left_result: ", left_result)
        right_result = Traceback(selection_matrix, q, j, 0, 0)
        # print("0-1 right_result: ", right_result)
        current_dep = merge_dict(left_result, right_result)

    elif d == 1 and c == 0:
        left_result = Traceback(selection_matrix, i, q, 1, 1)
        # print("1-0 left_result: ", left_result)
        right_result = Traceback(selection_matrix, q + 1, j, 0, 1)
        # print("1-0 right_result: ", right_result)
        current_dep = merge_dict(left_result, right_result)
        current_dep = merge_dict(current_dep, {i: [j]})
    elif d == 0 and c == 0:
        left_result = Traceback(selection_matrix, i, q, 1, 1)
        # print("0-0 left_result: ", left_result)
        right_result = Traceback(selection_matrix, q + 1, j, 0, 1)
        # print("0-0 right_result: ", right_result)
        current_dep = merge_dict(left_result, right_result)
        current_dep = merge_dict(current_dep, {j: [i]})

    return current_dep


def merge_dict(dict1, dict2):
    for k in dict2.keys():
        if k in dict1.keys():
            dict1[k].extend(dict2[k])
        else:
            dict1[k] = dict2[k]
    return dict1


# Doubly linked binary consituency Tree definition
class Node(object):
    def __init__(self, idx, nuclearity):
        self.idx = idx
        self.nuclearity = nuclearity
        self.parent = None
        self.children = [None, None]

    def add_child(self, child, branch):
        child.parent = self
        self.children[branch] = child


# Doubly linked dependency Tree definition
class Dep_Node(object):
    def __init__(self, idx):
        self.idx = idx
        self.parent = None
        self.children = []

    def add_child(self, child):
        child.parent = self
        self.children.append(child)


# Transform brackets to in-memory tree
def generate_const_tree(bracket_file):
    nodes_stack = []
    for node in bracket_file:
        node_index_tupel = node[0]
        node_nuclearity = node[1]
        # Check if node is a leaf
        if node_index_tupel[0] == node_index_tupel[1]:
            nodes_stack.append(Node(idx=node_index_tupel,
                                    nuclearity=node_nuclearity))
        # Add children for internal nodes
        else:
            tmp_node = Node(idx=node_index_tupel, nuclearity=node_nuclearity)
            tmp_node.add_child(nodes_stack.pop(), branch=1)
            tmp_node.add_child(nodes_stack.pop(), branch=0)
            nodes_stack.append(tmp_node)
    # Join last two nodes in root
    root_node = Node(idx='Root', nuclearity='Root')
    root_node.branch = 0
    root_node.add_child(nodes_stack.pop(), branch=1)
    root_node.add_child(nodes_stack.pop(), branch=0)
    #     print("POT")
    #     post_order_traversal(root_node)
    return root_node


def post_order_traversal(node):
    if node == None:
        return
    else:
        post_order_traversal(node.children[0])
        post_order_traversal(node.children[1])
    print(node.idx)


def const_to_dep_tree_li14(node):
    const_leaves = get_leaf_nodes(node)
    dep_tree = {}
    for leaf in const_leaves:
        p = find_top_node(leaf)
        if p.nuclearity == 'Root':
            root = leaf.idx[0]
        else:
            head = find_head_edu(p.parent)
            if head not in dep_tree:
                dep_tree[head] = []
            dep_tree[head].append(leaf.idx[0])
    return dep_tree


def find_top_node(e):
    C = e
    p = C.parent
    nucleus_child = [p.children[i] for i in range(len(p.children)) if p.children[i].nuclearity == 'Nucleus']
    while nucleus_child[0] == C and not p.nuclearity == 'Root':
        C = p
        p = C.parent
        nucleus_child = [p.children[i] for i in range(len(p.children)) if p.children[i].nuclearity == 'Nucleus']
    if p.nuclearity == 'Root' and nucleus_child[0] == C:
        C = p
    return C


def find_head_edu(p):
    while p.children != [None, None]:
        nucleus_child = [p.children[i] for i in range(len(p.children)) if p.children[i].nuclearity == 'Nucleus']
        p = nucleus_child[0]
    return p.idx[0]


def get_leaf_nodes(node):
    if node.children == [None, None]:
        return [node]
    else:
        leaves = []
        for child_node in node.children:
            leaves.extend(get_leaf_nodes(child_node))
        return leaves


def calculate_UAS(gold_tree, pred_tree):
    overlap = len(list(set(gold_tree) & set(pred_tree)))
    samples = len(gold_tree)
    return {"eisner_matches": overlap, "eisner_samples": samples}


def calculate_confident_score(pred_tree, edu_mat, dataname):
    # confidence score calculation C=averaged attn score of predicted decisions d
    decisions = [d.split('-') for d in pred_tree]
    attn_scores = 0.0
    for d in decisions:
        if dataname in ['gumconv', 'rst']:  # offset for gum
            d[0] = int(d[0]) - 1
            d[1] = int(d[1]) - 1
        attn_scores += edu_mat[int(d[0]), int(d[1])]
    confi_score = round((attn_scores / len(decisions)).item(), 4)
    return {"confident_score": confi_score}


def exec_eisner(curr_file, matrix, base_save_path, model_extension, layer, head, dataname, add_confi):
    filename = os.path.basename(curr_file).split(".")[0]
    # Skip if exists
    if os.path.exists(os.path.join(base_save_path, model_extension, "eisner_unlabelled_attachments",
                                   f"{layer}_{head}_{filename}.brackets")):
        with open(os.path.join(base_save_path, model_extension, "eisner_unlabelled_attachments",
                               f"{layer}_{head}_{filename}.brackets"), "r") as f:
            output = f.read().split("\n")
    else:
        output = stringify(eisner(matrix), 0)
        with open(os.path.join(base_save_path, model_extension, "eisner_unlabelled_attachments",
                               f"{layer}_{head}_{filename}.brackets"), "w") as f:
            f.write("\n".join(output))

    # Comparison
    # TODO: gold_tree is a dictionary with key = file_id, value = {1: [2,3,6], 2: [3], ...} where keys are head index, elements in list are dependents index
    gold_tree = GOLD_TREE[curr_file]
    gold_tree = stringify(gold_tree, 0)
    ret = calculate_UAS(gold_tree, output)

    if add_confi:  # li: add avg attn scores as confi score
        confi_score = calculate_confident_score(output, matrix, dataname)
        ret.update(confi_score)

    return ret





###Model Options
BART_large = {"d_name": "BART", "name":"facebook/bart-large", "prefix_len":1, "postfix_len":1, "max_input_size":1024, "batch_size": 1,
              "start_token":0, "end_token":2, "has_token_type_ids": False, "attn_keyword": "encoder_attentions", "model": BartModel}
BART_large_cnn = {"d_name": " + CNN", "name":"facebook/bart-large-cnn", "prefix_len":1, "postfix_len":1, "max_input_size":1024, "batch_size": 1,
            "start_token":0, "end_token":2, "has_token_type_ids": False, "attn_keyword": "encoder_attentions", "model": BartModel}
BART_large_samsum = {"d_name": " + SAMSUM", "name":"linydub/bart-large-samsum", "prefix_len":1, "postfix_len":1, "max_input_size":1024, "batch_size": 1,
            "start_token":0, "end_token":2, "has_token_type_ids": False, "attn_keyword": "encoder_attentions", "model": BartModel}
BART_large_squad2 = {"d_name": " + SQuAD2", "name":"phiyodr/bart-large-finetuned-squad2", "prefix_len":1, "postfix_len":1, "max_input_size":1024, "batch_size": 1,
            "start_token":0, "end_token":2, "has_token_type_ids": False, "attn_keyword": "encoder_attentions", "model": BartModel}
