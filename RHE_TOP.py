import torch
import copy
from transformers import BartModel
from RheModel import single_execution


def RheInfer(device, tokenizer, rhemodel, text):
    edu_lengths = []
    inputs = []
    slice_list = []

    for line in text:
        line = line.strip().lower().replace("\n", "").replace("\r", "") + " "
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

    curr_core = 0
    layer = 12
    head = range(16)
    model_params = {"d_name": " + CNN", "name": "facebook/bart-large-cnn", "prefix_len": 1, "postfix_len": 1,
                    "max_input_size": 1024, "batch_size": 1,
                    "start_token": 0, "end_token": 2, "has_token_type_ids": False,
                    "attn_keyword": "encoder_attentions", "model": BartModel}

    rhe_matrix_list = single_execution(device, rhemodel, model_params, layer, head, data_objects, len(inputs),
                                       edu_lengths)

    rhe_matrix = torch.zeros_like(rhe_matrix_list[0])
    for matrix in rhe_matrix_list:
        matrix_l2_norm = torch.norm(matrix)
        norm_matrix = matrix / matrix_l2_norm
        rhe_matrix = rhe_matrix + norm_matrix
    rhe_matrix = rhe_matrix / len(rhe_matrix_list)
    ###存疑
    rhe_matrix = rhe_matrix.to(device)
    for s1 in range(rhe_matrix.size(0)):
        rhe_matrix[s1, s1] = 0.0

    e_matrix1 = rhe_matrix.clone().detach().to(device)

    # 对e_matrix进行【0，1】缩放
    e_matrix = torch.zeros_like(e_matrix1).to(device)
    e_rows, e_cols = torch.triu_indices(e_matrix1.size(0), e_matrix1.size(1), offset=1)
    upper_half_e = e_matrix1[e_rows, e_cols]
    for i in range(e_matrix1.size(0) - 1):
        for j in range(i + 1, e_matrix1.size(1)):
            e_matrix[i, j] = (e_matrix1[i, j] - upper_half_e.min()) / (upper_half_e.max() - upper_half_e.min())

    return rhe_matrix, e_matrix