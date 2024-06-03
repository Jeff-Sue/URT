import torch
import bisect
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel
from transformers.models.bert.modeling_bert import *
from transformers import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss


class MarginRankingLoss():
    def __init__(self, margin):
        self.margin = margin

    def __call__(self, p_scores, n_scores):
        scores = self.margin - (p_scores - n_scores)
        scores = scores.clamp(min=0)

        return scores.mean()


class SegModel(nn.Module):
    def __init__(self, args, model_path='', margin=1, train_split=5, window_size=5):
        super(SegModel, self).__init__()
        self.args = args
        self.margin = margin
        self.train_split = train_split
        self.window_size = window_size
        self.topic_model = AutoModel.from_pretrained(model_path + "princeton-nlp/sup-simcse-bert-base-uncased")
        self.coheren_model = BertForNextSentencePrediction.from_pretrained(model_path + "bert-base-uncased",
                                                                           num_labels=2,
                                                                           output_attentions=False,
                                                                           output_hidden_states=True,
                                                                           args=args)

        self.topic_loss = nn.CrossEntropyLoss()
        self.score_loss = MarginRankingLoss(self.margin)

    def forward(self, input_data, window_size=None):
        print("TOP forward is used!.")
        device, topic_loss = input_data['coheren_inputs'].device, torch.tensor(0)
        topic_context_count, topic_pos_count, topic_neg_count = 0, 0, 0
        topic_context_mean, topic_pos_mean, topic_neg_mean = [], [], []

        coheren_pos_scores = self.coheren_model(lengths1=input_data['coheren_inputs'][:, 0, :],
                                                                     attention_mask=input_data['coheren_mask'][:, 0, :],
                                                                     token_type_ids=input_data['coheren_type'][:, 0, :])[0]
        coheren_neg_scores = self.coheren_model(input_data['coheren_inputs'][:, 1, :],
                                                                     attention_mask=input_data['coheren_mask'][:, 1, :],
                                                                     token_type_ids=input_data['coheren_type'][:, 1, :])[0]

        batch_size = len(input_data['topic_context_num'])
        topic_context = self.topic_model(input_data['topic_context'], input_data['topic_context_mask'])[1]
        topic_pos = self.topic_model(input_data['topic_pos'], input_data['topic_pos_mask'])[1]
        topic_neg = self.topic_model(input_data['topic_neg'], input_data['topic_neg_mask'])[1]

        topic_loss = self.topic_train(input_data, window_size)

        for i, j, z in zip(input_data['topic_context_num'], input_data['topic_pos_num'], input_data['topic_neg_num']):
            topic_context_mean.append(torch.mean(topic_context[topic_context_count:topic_context_count + i], dim=0))
            topic_pos_mean.append(torch.mean(topic_pos[topic_pos_count:topic_pos_count + j], dim=0))
            topic_neg_mean.append(torch.mean(topic_neg[topic_neg_count:topic_neg_count + z], dim=0))
            topic_context_count, topic_pos_count, topic_neg_count = topic_context_count + i, topic_pos_count + j, topic_neg_count + z

        assert len(topic_context_mean) == len(topic_pos_mean) == len(topic_neg_mean) == batch_size

        topic_context_mean, topic_pos_mean = pad_sequence(topic_context_mean, batch_first=True), pad_sequence(
            topic_pos_mean, batch_first=True)
        topic_neg_mean = pad_sequence(topic_neg_mean, batch_first=True)

        topic_pos_scores = F.cosine_similarity(topic_context_mean, topic_pos_mean, dim=1,
                                               eps=1e-08).to(device)
        topic_neg_scores = F.cosine_similarity(topic_context_mean, topic_neg_mean, dim=1,
                                               eps=1e-08).to(device)

        pos_scores = coheren_pos_scores[:, 0] + topic_pos_scores
        neg_scores = coheren_neg_scores[:, 0] + topic_neg_scores

        margin_loss = self.score_loss(pos_scores, neg_scores)

        loss = margin_loss.clone() + topic_loss
        return loss, margin_loss, topic_loss

    def infer(self, lengths1, lengths2, coheren_input, coheren_mask, coheren_type_id, topic_input=None, topic_mask=None, topic_num=None):
        device = coheren_input.device
        torch.cuda.empty_cache()

        batch_size = 100
        total_samples = coheren_input.size(0)
        num_batches = (total_samples + batch_size - 1) // batch_size
        coheren_scores_list = []
        for i in range(num_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, total_samples)

            # 选择当前批次的数据
            coheren_input_batch = coheren_input[start_idx:end_idx]
            coheren_mask_batch = coheren_mask[start_idx:end_idx]
            coheren_type_id_batch = coheren_type_id[start_idx:end_idx]
            lengths1_batch = lengths1[start_idx:end_idx]
            lengths2_batch = lengths2[start_idx:end_idx]

            coheren_scores_batch = self.coheren_model(lengths1_batch, lengths2_batch, coheren_input_batch, coheren_mask_batch, coheren_type_id_batch)[0]
            del coheren_input_batch, coheren_mask_batch, coheren_type_id_batch, lengths1_batch, lengths2_batch
            coheren_scores_list.append(coheren_scores_batch)
        coheren_scores = torch.cat(coheren_scores_list, dim=0)

        del coheren_input, coheren_mask, coheren_type_id, lengths1, lengths2
        # coheren_scores = self.coheren_model(lengths1, lengths2, coheren_input, coheren_mask, coheren_type_id)[0]


        chunk_size = 200
        chunks_input0 = [topic_input[0][i:i+chunk_size, :] for i in range(0, topic_input[0].size(0), chunk_size)]
        chunks_input1 = [topic_input[1][i:i + chunk_size, :] for i in range(0, topic_input[1].size(0), chunk_size)]
        chunks_mask0 = [topic_mask[0][i:i + chunk_size, :] for i in range(0, topic_mask[0].size(0), chunk_size)]
        chunks_mask1 = [topic_mask[1][i:i + chunk_size, :] for i in range(0, topic_mask[1].size(0), chunk_size)]

        topic_contexts = []
        topic_curs = []

        for i in range(len(chunks_input0)):
            cur_context = self.topic_model(chunks_input0[i], chunks_mask0[i])[1]
            topic_contexts.append(cur_context)
            torch.cuda.empty_cache()
        for i in range(len(chunks_input0)):
            cur_cur = self.topic_model(chunks_input1[i], chunks_mask1[i])[1]
            topic_curs.append(cur_cur)
            torch.cuda.empty_cache()

        topic_context = torch.cat(topic_contexts, 0).to(device)
        topic_cur = torch.cat(topic_curs, 0).to(device)

        topic_context_count = topic_cur_count = 0
        topic_context_mean, topic_cur_mean = [], []
        for i, j in zip(topic_num[0], topic_num[1]):
            topic_context_mean.append(torch.mean(topic_context[topic_context_count:topic_context_count + i], dim=0))
            topic_cur_mean.append(torch.mean(topic_cur[topic_cur_count:topic_cur_count + j], dim=0))
            topic_context_count, topic_cur_count = topic_context_count + i, topic_cur_count + j
        topic_context_mean, topic_cur_mean = pad_sequence(topic_context_mean, batch_first=True), pad_sequence(
            topic_cur_mean, batch_first=True)
        topic_scores = F.cosine_similarity(topic_context_mean, topic_cur_mean, dim=1, eps=1e-08).to(device)

        max_val = topic_scores.max()  # 获取张量的最大值
        min_val = topic_scores.min()  # 获取张量的最小值
        if max_val == min_val:
            # 防止出现全是1报错
            scaled_topic = topic_scores
        else:
            # scale = 2 / (max_val - min_val)  # 计算缩放比例
            # scaled_topic = (topic_scores - min_val) * scale - 1  # 进行缩放操作
            mean = topic_scores.mean()
            std = topic_scores.std()
            scaled_topic = (topic_scores - mean) / std

        max_val = coheren_scores[:, 0].max()  # 获取张量的最大值
        min_val = coheren_scores[:, 0].min()  # 获取张量的最小值
        mean = coheren_scores[:, 0].mean()
        std = coheren_scores[:, 0].std()
        scaled_tensor = 2 * (coheren_scores[:, 0] - mean) / std
        final_scores = scaled_tensor + scaled_topic

        return torch.sigmoid(final_scores)

    def topic_train(self, input_data, window_size):
        device, batch_size = input_data['coheren_inputs'].device, len(input_data['topic_context_num'])
        topic_all = self.topic_model(input_data['topic_train'], input_data['topic_train_mask'])[1]
        true_segments, segments, neg_utts, seg_num, count, margin_count, topic_loss = [], [], [], [], 0, batch_size, torch.tensor(
            0).to(device, dtype=torch.float)

        # pseudo-segmentation
        for b in range(batch_size):
            cur_num = input_data['topic_num'][b]
            dial_len, cur_utt = cur_num[0], cur_num[1]
            cur = topic_all[count:count + dial_len]
            assert dial_len > cur_utt

            top_cons, top_curs = [], []
            for i in range(1, dial_len):
                top_con = torch.mean(cur[max(0, i - 2): i], dim=0)
                top_cur = torch.mean(cur[i: min(dial_len, i + 2)], dim=0)
                top_cons.append(top_con)
                top_curs.append(top_cur)

            top_cons, top_curs = pad_sequence(top_cons, batch_first=True), pad_sequence(top_curs, batch_first=True)
            topic_scores = F.cosine_similarity(top_cons, top_curs, dim=1, eps=1e-08).to(device)
            depth_scores = tet(torch.sigmoid(topic_scores))
            tet_seg = np.argsort(np.array(depth_scores))[-self.train_split:] + 1
            tet_seg = [0] + tet_seg.tolist() + [dial_len]
            tet_seg.sort()

            tet_mid = bisect.bisect(tet_seg, cur_utt)
            tet_mid_seg = (tet_seg[tet_mid - 1], tet_seg[tet_mid])

            pos_left = max(tet_mid_seg[0], cur_utt - window_size)
            pos_right = min(tet_mid_seg[1], cur_utt + window_size + 1)

            neg_left = min(tet_seg[max(0, tet_mid - 1)], cur_utt - window_size)
            neg_right = max(tet_seg[tet_mid], cur_utt + window_size + 1)

            mid = torch.mean(cur[pos_left:pos_right], dim=0).unsqueeze(0)
            segments.append([[mid], (pos_left, pos_right), count, (neg_left, neg_right)])

            count += dial_len

        # Margin loss
        for b in range(batch_size):
            cur_seg = segments[b]
            mid_left, mid_right = cur_seg[1]
            neg_left, neg_right = cur_seg[3]
            count, cur_num = cur_seg[2], input_data['topic_num'][b]
            dial_len, cur_utt, mid_seg = cur_num[0], cur_num[1], cur_seg[0][0]

            neg = torch.cat((topic_all[:count + neg_left], topic_all[count + neg_right:]), dim=0)

            anchor = topic_all[count + cur_utt].unsqueeze(0)
            pos = torch.cat(
                (topic_all[count + mid_left:count + cur_utt], topic_all[count + cur_utt + 1:count + mid_right]), dim=0)
            pos_score = F.cosine_similarity(anchor, pos, dim=1)

            if pos_score.shape[0] == 0:
                margin_count -= 1
                continue

            neg_score = F.cosine_similarity(anchor, neg, dim=1)
            margin_pos = pos_score.unsqueeze(0).repeat(neg_score.shape[0], 1).T.flatten()
            margin_neg = neg_score.repeat(pos_score.shape[0])
            assert margin_pos.shape == margin_neg.shape
            cur_loss = self.score_loss(margin_pos, margin_neg)
            if torch.isnan(cur_loss):
                print('Encounter nan:', pos_score.shape, neg_score.shape)
                margin_count -= 1
                continue
            topic_loss += cur_loss

        topic_loss /= margin_count
        return topic_loss


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertForNextSentencePrediction(BertPreTrainedModel):
    def __init__(self, config, args):
        super().__init__(config)

        self.bert = BertModel(config)
        self.cls = BertOnlyNSPHead(config)
        self.args = args
        self.init_weights()

    def forward(
        self,
        length1=None,
        length2=None,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        next_sentence_label=None,
    ):

        r"""
        next_sentence_label (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the next sequence prediction (classification) loss. Input should be a sequence pair (see ``input_ids`` docstring)
            Indices should be in ``[0, 1]``.
            ``0`` indicates sequence B is a continuation of sequence A,
            ``1`` indicates sequence B is a random sequence.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`next_sentence_label` is provided):
            Next sequence prediction (classification) loss.
        seq_relationship_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertTokenizer, BertForNextSentencePrediction
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertForNextSentencePrediction.from_pretrained('bert-base-uncased')

        prompt = "In Italy, pizza served in formal settings, such as at a restaurant, is presented unsliced."
        next_sentence = "The sky is blue due to the shorter wavelength of blue light."
        encoding = tokenizer.encode_plus(prompt, next_sentence, return_tensors='pt')

        loss, logits = model(**encoding, next_sentence_label=torch.LongTensor([1]))
        assert logits[0, 0] < logits[0, 1] # next sentence was random
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        hs_1 = torch.empty(outputs[0].size(0), outputs[0].size(2)).to(length1.device)
        hs_2 = torch.empty(outputs[0].size(0), outputs[0].size(2)).to(length2.device)
        for i in range(outputs[0].size(0)):
            hs_1[i] = outputs[0][i, :min(self.args.max_text_length-1, int(length1[i]))].mean(dim=0)

        for i in range(outputs[0].size(0)):
            hs_2[i] = outputs[0][i, min(self.args.max_text_length-1, int(length1[i])):min(self.args.max_text_length, int(length1[i]) + int(length2[i]))].mean(dim=0)
            # print(outputs[0].size(), int(length1[i]) + int(length2[i]))
        # hs = torch.cat((hs_1, hs_2), dim=1)
        hs = (hs_1 + hs_2)/2

        # pooled_output = outputs[1]
        seq_relationship_score = self.cls(hs)

        outputs = (seq_relationship_score,) + outputs[2:]  # add hidden states and attention if they are here
        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            outputs = (next_sentence_loss,) + outputs

        return outputs  # (next_sentence_loss), seq_relationship_score, (hidden_states), (attentions)


def tet(scores):
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
        output_scores.append(depth_score.cpu().detach())

    return output_scores