import random
import time

import numpy as np
import torch
from nltk.parse.dependencygraph import DependencyGraph

# Some constants
L, R = 0, 1
I, C = 0, 1
DIRECTIONS = (L, R)
COMPLETENESS = (I, C)
NEG_INF = -float('inf')


class Span(object):
    def __init__(self, left_idx, right_idx, head_side, complete):
        self.data = (left_idx, right_idx, head_side, complete)

    @property
    def left_idx(self):
        return self.data[0]

    @property
    def right_idx(self):
        return self.data[1]

    @property
    def head_side(self):
        return self.data[2]

    @property
    def complete(self):
        return self.data[3]

    def __str__(self):
        return "({}, {}, {}, {})".format(
            self.left_idx,
            self.right_idx,
            "L" if self.head_side == L else "R",
            "C" if self.complete == C else "I",
        )

    def __repr__(self):
        return self.__str__()

    def __hash__(self):
        return hash(self.data)

    def __eq__(self, other):
        return isinstance(other, Span) and hash(other) == hash(self)

def eisner2matrix(weight):
    """
    `N` denotes the length of sentence.

    :param weight: size N x N
    :return: the projective tree with maximum score
    """
    N = weight.size(0)

    btp = {}  # Back-track pointer
    dp_s = {}

    # Init
    for i in range(N):
        for j in range(i + 1, N):
            for dir in DIRECTIONS:
                for comp in COMPLETENESS:
                    dp_s[Span(i, j, dir, comp)] = NEG_INF

    # base case
    for i in range(N):
        for dir in DIRECTIONS:
            dp_s[Span(i, i, dir, C)] = 0.
            btp[Span(i, i, dir, C)] = None

    rules = [
        # span_shape_tuple := (span_direction, span_completeness),
        # rule := (span_shape, (left_subspan_shape, right_subspan_shape))
        ((L, I), ((R, C), (L, C))),
        ((R, I), ((R, C), (L, C))),
        ((L, C), ((L, C), (L, I))),
        ((R, C), ((R, I), (R, C))),
    ]
    num = 0
    weight_matrix = torch.zeros_like(weight).to(weight.device)
    num_matrix = torch.zeros_like(weight).to(weight.device)
    matrix = torch.zeros_like(weight).to(weight.device)
    for size in range(1, N):
        for i in range(0, N - size):
            j = i + size
            for rule in rules:
                ((dir, comp), ((l_dir, l_comp), (r_dir, r_comp))) = rule

                if comp == I:
                    edge_w = weight[i, j] if (dir == R) else weight[j, i]
                    k_start, k_end = (i, j)
                    offset = 1
                else:
                    edge_w = 0.
                    k_start, k_end = (i + 1, j + 1) if dir == R else (i, j)
                    offset = 0

                span = Span(i, j, dir, comp)
                all_w = torch.tensor(0.0).to(weight.device)
                dp_s2 = dp_s
                for k in range(k_start, k_end):
                    l_span = Span(i, k, l_dir, l_comp)
                    r_span = Span(k + offset, j, r_dir, r_comp)
                    s = edge_w + dp_s2[l_span] + dp_s2[r_span]
                    if s > -1:
                        if l_span.complete == I:
                            if l_span.head_side == R:
                                all_w += s
                        if r_span.complete == I:
                            if r_span.head_side == R:
                                all_w += s
                    if s > dp_s2[span]:
                        dp_s2[span] = s

                for k in range(k_start, k_end):
                    l_span = Span(i, k, l_dir, l_comp)
                    r_span = Span(k + offset, j, r_dir, r_comp)
                    s = edge_w + dp_s[l_span] + dp_s[r_span]
                    num += 1
                    ### 防止s为-inf
                    if s > -1:
                        if l_span.complete == I:
                            if l_span.head_side == R:
                                num_matrix[l_span.left_idx, l_span.right_idx] += 1.0
                                weight_matrix[l_span.left_idx, l_span.right_idx] += s / all_w
                        if r_span.complete == I:
                            if r_span.head_side == R:
                                num_matrix[r_span.left_idx, r_span.right_idx] += 1.0
                                weight_matrix[r_span.left_idx, r_span.right_idx] += s / all_w
                    if s > dp_s[span]:
                        dp_s[span] = s
    print("weight_matrix: ", weight_matrix)
    print("num_matrix: ", num_matrix)
    for s1 in range(weight_matrix.size(0) - 1):
        for s2 in range(s1+1, weight_matrix.size(0)):
            if num_matrix[s1, s2] != 0:
                matrix[s1, s2] = weight_matrix[s1, s2] / num_matrix[s1, s2]
    # recover tree
    return matrix

def eisner(weight):
    """
    `N` denotes the length of sentence.

    :param weight: size N x N
    :return: the projective tree with maximum score
    """
    N = weight.size(0)

    btp = {}  # Back-track pointer
    dp_s = {}

    # Init
    for i in range(N):
        for j in range(i + 1, N):
            for dir in DIRECTIONS:
                for comp in COMPLETENESS:
                    dp_s[Span(i, j, dir, comp)] = NEG_INF

    # base case
    for i in range(N):
        for dir in DIRECTIONS:
            dp_s[Span(i, i, dir, C)] = 0.
            btp[Span(i, i, dir, C)] = None

    rules = [
        # span_shape_tuple := (span_direction, span_completeness),
        # rule := (span_shape, (left_subspan_shape, right_subspan_shape))
        ((L, I), ((R, C), (L, C))),
        ((R, I), ((R, C), (L, C))),
        ((L, C), ((L, C), (L, I))),
        ((R, C), ((R, I), (R, C))),
    ]
    num = 0
    for size in range(1, N):
        for i in range(0, N - size):
            j = i + size
            for rule in rules:
                ((dir, comp), ((l_dir, l_comp), (r_dir, r_comp))) = rule

                if comp == I:
                    edge_w = weight[i, j] if (dir == R) else weight[j, i]
                    k_start, k_end = (i, j)
                    offset = 1
                else:
                    edge_w = 0.
                    k_start, k_end = (i + 1, j + 1) if dir == R else (i, j)
                    offset = 0

                span = Span(i, j, dir, comp)
                for k in range(k_start, k_end):
                    l_span = Span(i, k, l_dir, l_comp)
                    r_span = Span(k + offset, j, r_dir, r_comp)
                    s = edge_w + dp_s[l_span] + dp_s[r_span]
                    num += 1
                    if s > dp_s[span]:
                        dp_s[span] = s
                        btp[span] = (l_span, r_span)

    # recover tree
    return back_track(btp, Span(0, N - 1, R, C), set())

def back_track(btp, span, edge_set):
    if span.complete == I:
        if span.head_side == L:
            edge = (span.right_idx, span.left_idx)
        else:
            edge = (span.left_idx, span.right_idx)
        edge_set.add(edge)

    if btp[span] is not None:
        l_span, r_span = btp[span]

        back_track(btp, l_span, edge_set)
        back_track(btp, r_span, edge_set)
    else:
        return

    return edge_set

def edges_to_dg(edge_set, words):
    N = len(edge_set)
    dg = DependencyGraph()
    for i in range(1, N + 1):
        dg.add_node({'address': i, 'word': words[i - 1]})
    for (h, c) in edge_set:
        dg.add_arc(h, c)
    dg.nodes[0]['word'] = 'ROOT'
    return dg

def test_case_1():
    weight = np.array([[0.0076, 0.0079, 0.0077, 0.0077, 0.0076, 0.0085, 0.0080, 0.0078, 0.0078,
                        0.0077, 0.0079, 0.0061],
                       [0.0000, 0.0078, 0.0079, 0.0080, 0.0076, 0.0077, 0.0078, 0.0083, 0.0081,
                        0.0076, 0.0087, 0.0062],
                       [0.0000, 0.0000, 0.0076, 0.0083, 0.0077, 0.0080, 0.0075, 0.0078, 0.0078,
                        0.0080, 0.0080, 0.0059],
                       [0.0000, 0.0000, 0.0000, 0.0077, 0.0081, 0.0089, 0.0077, 0.0080, 0.0079,
                        0.0079, 0.0083, 0.0059],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0077, 0.0086, 0.0077, 0.0079, 0.0078,
                        0.0079, 0.0077, 0.0064],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0080, 0.0079, 0.0075, 0.0078,
                        0.0082, 0.0079, 0.0056],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0076, 0.0078, 0.0079,
                        0.0079, 0.0083, 0.0056],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0077, 0.0077,
                        0.0081, 0.0078, 0.0054],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0078,
                        0.0080, 0.0081, 0.0057],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0078, 0.0080, 0.0061],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0080, 0.0062],
                       [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.0059]])

    return eisner(weight)

# result = edges_to_dg(test_case_1(), ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L'])
#
# rels = {}
# for i in result.nodes:
#     v = result.nodes[i]['deps'].values()
#     if len(v) > 0:
#         rels[i] = list(v)[0]
# print(rels)
### 计算下一轮的span和权重值
def Linearization(span, w, weight_matrix):
    s = []
    ws = []
    if span.complete == C and span.head_side == R:
        for i in range(span.left_idx + 1, span.right_idx + 1):
            if span.left_idx != i:
                s.append(Span(span.left_idx, i, R, I))
                ws.append(w * weight_matrix[span.left_idx, i])
            if span.right_idx != i:
                s.append(Span(i, span.right_idx, R, C))
                ws.append(w * weight_matrix[i, span.right_idx])
    if span.complete == I and span.head_side == R:
        for i in range(span.left_idx + 1, span.right_idx + 1):
            if span.left_idx != i:
                s.append(Span(span.left_idx, i, R, C))
                ws.append(w * weight_matrix[span.left_idx, i])
            if span.right_idx != i:
                s.append(Span(i, span.right_idx, L, C))
                ws.append(w * weight_matrix[i, span.right_idx])

    return s, ws


def eisner_matrix(weight):
    eisner_weight = torch.zeros_like(weight).to(weight.device)
    start_span = [Span(0, weight.size(0)-1, R, C)]
    ws = [1]
    ### 设置前5次作为取值
    for k in range(weight.size(0)):
        start_span_news = []
        ws_news = []
        for i in range(len(start_span)):

            start_span_new, ws_new = Linearization(start_span[i], ws[i], weight)
            start_span_news.extend(start_span_new)
            ws_news.extend(ws_new)
            # for j in range(len(start_span_new)):
            #     if start_span_new[j].complete == I and start_span_new[j].head_side == R:
            #         eisner_weight[start_span_new[j].left_idx, start_span_new[j].right_idx] += ws_new[j]
        random.seed(42)
        if len(start_span_news) > int(500/weight.size(0)):
            ws = sorted(ws_news, reverse=True)[:int(500/weight.size(0))]
            indices = [ws_news.index(w) for w in ws]
            start_span = [start_span_news[idx] for idx in indices]
        else:
            start_span = start_span_news
            ws = ws_news

        value_dict = get_value(start_span, ws)
        for key, value in value_dict.items():
            eisner_weight[key.left_idx, key.right_idx] += value
    eisner_weight = eisner_weight / torch.max(eisner_weight)

    return eisner_weight


### 对一轮的span结果进行统计和概率计算
def get_value(spans, ws):
    value_dict = {}
    for i in range(len(spans)):
        if spans[i] not in value_dict.keys():
            value_dict[spans[i]] = ws[i]
        else:
            value_dict[spans[i]] += ws[i]
    values = 0.0
    for key, value in value_dict.items():
        values += value
    for key, value in value_dict.items():
        value_dict[key] = value / values

    return value_dict


def prim_mst(graph):

    num_vertices = len(graph)

    # 初始化最小生成树的顶点集合和边集合

    mst = []

    visited = [False] * num_vertices


    # 选择初始顶点
    start_vertex = 0

    visited[start_vertex] = True



    while len(mst) < num_vertices - 1:

        min_weight = float('inf')

        min_edge = None
        # 遍历已访问顶点集合，寻找权重最小的边
        for i in range(num_vertices):

            if visited[i]:

                for j in range(num_vertices):

                    if not visited[j] and graph[i][j] < min_weight:
                        min_weight = graph[i][j]

                        min_edge = (i, j)



        if min_edge:

            # 将权重最小的边加入最小生成树

            mst.append(min_edge)

            visited[min_edge[1]] = True

    return mst