import torch
from torch.nn.utils.rnn import pad_sequence
from keras.preprocessing.sequence import pad_sequences
import numpy as np


def TopSingle2HC(args, tokenizer, text):
    single_topic_input1 = []
    single_topic_input2 = []
    single_topic_attention1 = []
    single_topic_attention2 = []
    single_topic_num1 = []
    single_topic_num2 = []
    single_coheren_input1 = []
    single_coheren_input2 = []
    type_ids = []
    id_inputs = []
    coheren_att_masks = []
    c_id_inputs = []
    c_type_ids = []
    c_coheren_att_masks = []
    lengths1 = []
    lengths2 = []

    for i in range(len(text) - 1):
        context, cur = [], []
        l, r = i, i + 1
        for win in range(args.window_size):
            if l > -1:
                context.append(text[l][:128])
                l -= 1
            if r < len(text):
                cur.append(text[r][:128])
                r = r + 1
        context.reverse()

        topic_con = tokenizer(context, truncation=True, padding=True, max_length=256, return_tensors='pt')
        topic_cur = tokenizer(cur, truncation=True, padding=True, max_length=256, return_tensors='pt')
        single_topic_input1.append(topic_con['input_ids'])
        single_topic_input2.append(topic_cur['input_ids'])
        single_topic_attention1.append(topic_con['attention_mask'])
        single_topic_attention2.append(topic_cur['attention_mask'])
        single_topic_num1.append(len(context))
        single_topic_num2.append(len(cur))

        sent1 = ''
        for sen in context:
            sent1 = sent1 + sen + '[SEP]'

        sent2 = text[i + 1]

        encoded_sent1 = tokenizer.encode(sent1, truncation=True, padding=True, add_special_tokens=True,
                                         max_length=256,
                                         return_tensors='pt')
        encoded_sent2 = tokenizer.encode(sent2, truncation=True, padding=True, add_special_tokens=True,
                                         max_length=256,
                                         return_tensors='pt')

        encoded_pair = encoded_sent1[0].tolist()[:-1] + encoded_sent2[0].tolist()[1:]
        type_id = [0] * len(encoded_sent1[0].tolist()[:-1]) + [1] * len(encoded_sent2[0].tolist()[1:])
        type_ids.append(torch.Tensor(type_id))
        id_inputs.append(torch.Tensor(encoded_pair))

        single_coheren_input1.append(encoded_sent1)
        single_coheren_input2.append(encoded_sent2)

    MAX_LEN = args.max_text_length
    id_inputs = pad_sequences(id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    type_ids = pad_sequences(type_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")
    for sent in id_inputs:
        att_mask = [int(token_id > 0) for token_id in sent]
        coheren_att_masks.append(att_mask)

    single_topic_input1.append(topic_cur['input_ids'])
    single_topic_input2 = [topic_con['input_ids']] + single_topic_input2
    single_topic_attention1.append(topic_cur['attention_mask'])
    single_topic_attention2 = [topic_con['attention_mask']] + single_topic_attention2
    single_topic_num1.append(len(cur))
    single_topic_num2 = [len(context)] + single_topic_num2
    single_coheren_input1.append(encoded_sent2)
    single_coheren_input2 = [encoded_sent1] + single_coheren_input2

    for ci in range(len(text) - 1):
        for cj in range(ci + 1, len(text)):
            c_encoded_pair = single_coheren_input1[ci][0].tolist()[:-1][:99] + single_coheren_input2[cj][0].tolist()[
                                                                               1:][:99]
            lengths1.append(len(single_coheren_input1[ci][0].tolist()[:-1][:99]))
            lengths2.append(len(single_coheren_input2[cj][0].tolist()[1:][:99]))
            c_type_id = [0] * len(single_coheren_input1[ci][0].tolist()[:-1]) + [1] * len(
                single_coheren_input2[cj][0].tolist()[1:])
            c_type_ids.append(torch.Tensor(c_type_id))
            c_id_inputs.append(torch.Tensor(c_encoded_pair))

    MAX_LEN = args.max_text_length
    c_id_inputs = pad_sequences(c_id_inputs, maxlen=MAX_LEN, dtype="long", value=0, truncating="post", padding="post")
    c_type_ids = pad_sequences(c_type_ids, maxlen=MAX_LEN, dtype="long", value=1, truncating="post", padding="post")
    for sent in c_id_inputs:
        c_att_mask = [int(token_id > 0) for token_id in sent]
        c_coheren_att_masks.append(c_att_mask)

    del text, id_inputs, type_ids
    c_coheren_inputs = torch.tensor(c_id_inputs).to(args.device)
    c_coheren_masks = torch.tensor(c_coheren_att_masks).to(args.device)
    c_coheren_type_ids = torch.tensor(c_type_ids).to(args.device)
    lengths1 = torch.tensor(lengths1).to(args.device)
    lengths2 = torch.tensor(lengths2).to(args.device)

    return single_topic_input1, single_topic_input2, single_topic_attention1, single_topic_attention2, single_topic_num1, single_topic_num2, c_coheren_inputs, c_coheren_masks, c_coheren_type_ids, lengths1, lengths2


def TopHC2Matrix(args, Topmodel, text, single_topic_input1, single_topic_input2, single_topic_attention1, single_topic_attention2, single_topic_num1, single_topic_num2,  c_coheren_inputs, c_coheren_masks, c_coheren_type_ids, lengths1, lengths2):

    _, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[], []], [
        [], []], [[], []]
    depth_scores = []
    for i in range(len(text)):
        for j in range(i+1, len(text)):
            topic_input[0].extend(single_topic_input1[i])
            topic_input[1].extend(single_topic_input2[j])
            topic_att_mask[0].extend(single_topic_attention1[i])
            topic_att_mask[1].extend(single_topic_attention2[j])
            topic_num[0].append(single_topic_num1[i])
            topic_num[1].append(single_topic_num2[j])

    try:
        topic_input = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_input]
        topic_mask = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_att_mask]
    except:
        print(text)


    scores = Topmodel.infer(lengths1, lengths2, c_coheren_inputs, c_coheren_masks, c_coheren_type_ids, topic_input, topic_mask, topic_num)
    top_matrix = torch.zeros(len(text), len(text))
    num = 0
    for i in range(len(text)):
        for j in range(i+1, len(text)):
            top_matrix[i, j] = scores[num]
            num += 1
    #
    # depth_scores = self.depth_score_cal(scores)
    # print("depth_scores: ", len(depth_scores), depth_scores)

    return top_matrix


def HCExtraction(args, Topmodel, text, single_topic_input1, single_topic_input2, single_topic_attention1,
                 single_topic_attention2, single_topic_num1, single_topic_num2, c_coheren_inputs, c_coheren_masks, c_coheren_type_ids, lengths1, lengths2):

    # _, id_inputs, coheren_att_masks, type_ids, topic_input, topic_att_mask, topic_num = [], [], [], [], [[], []], [
    #     [], []], [[], []]

    topic_input = [[], []]
    topic_att_mask = [[], []]
    topic_num = [[], []]

    for i in range(len(text)):
        topic_input[0].extend(single_topic_input1[i])
        topic_input[1].extend(single_topic_input2[i])
        topic_att_mask[0].extend(single_topic_attention1[i])
        topic_att_mask[1].extend(single_topic_attention2[i])
        topic_num[0].append(single_topic_num1[i])
        topic_num[1].append(single_topic_num2[i])


    try:
        topic_input = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_input]
        topic_mask = [pad_sequence(i, batch_first=True).to(args.device) for i in topic_att_mask]
    except:
        print(text)

    coheren_feature = Topmodel.coheren_model.bert(c_coheren_inputs, c_coheren_masks, c_coheren_type_ids)[0]
    coheren_feature1 = torch.stack(
        [coheren_feature[i, :lengths1[i]].mean(dim=0) for i in range(coheren_feature.size(0))])
    coheren_feature2 = torch.stack([coheren_feature[i, lengths1[i]:lengths1[i] + lengths2[i]].mean(dim=0) for i in
                                    range(coheren_feature.size(0))])

    topic_context = Topmodel.topic_model(topic_input[0], topic_mask[0])[1]
    topic_cur = Topmodel.topic_model(topic_input[1], topic_mask[1])[1]
    del coheren_feature
    topic_context_mean = torch.stack(
        [torch.mean(topic_context[sum(topic_num[0][:i]):sum(topic_num[0][:i + 1])], dim=0) for i in
         range(len(topic_num[0]))])
    topic_cur_mean = torch.stack(
        [torch.mean(topic_cur[sum(topic_num[1][:i]):sum(topic_num[1][:i + 1])], dim=0) for i in
         range(len(topic_num[1]))])

    return topic_context_mean, topic_cur_mean, coheren_feature1, coheren_feature2


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


def compute_boundary(args, matrix):
    scores = []
    for i in range(matrix.size(0)-1):
        scores.append(matrix[i, i+1])
    output_scores = depth_score_cal(scores)
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