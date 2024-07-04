import json

# with open("Molweni_ori_result.json", 'r') as f1:
#     ori = json.load(f1)
#
# with open("Molweni_pred_result.json", 'r') as f2:
#     pred = json.load(f2)
#
# with open("Molweni_test_result.json", 'r') as f3:
#     test = json.load(f3)


def analysis1(ori, pred, test):
    ori_ana = {}
    pred_ana = {}
    for i in range(len(ori)):
        a1 = [l for l in ori[i] if l in test[i]]
        for j in a1:
            if abs(j[1] - j[0]) not in ori_ana.keys():
                ori_ana[abs(j[1] - j[0])] = 1
            else:
                ori_ana[abs(j[1] - j[0])] += 1

        a2 = [l for l in pred[i] if l in test[i]]
        for j in a2:
            if abs(j[1] - j[0]) not in pred_ana.keys():
                pred_ana[abs(j[1] - j[0])] = 1
            else:
                pred_ana[abs(j[1] - j[0])] += 1

    ori_ana2 = {}
    pred_ana2 = {}
    test_ana2 = {}
    for i in range(len(ori)):
        for j in ori[i]:
            if abs(j[1] - j[0]) not in ori_ana2.keys():
                ori_ana2[abs(j[1] - j[0])] = 1
            else:
                ori_ana2[abs(j[1] - j[0])] += 1

        for j in pred[i]:
            if abs(j[1] - j[0]) not in pred_ana2.keys():
                pred_ana2[abs(j[1] - j[0])] = 1
            else:
                pred_ana2[abs(j[1] - j[0])] += 1

        for j in test[i]:
            if abs(j[1] - j[0]) not in test_ana2.keys():
                test_ana2[abs(j[1] - j[0])] = 1
            else:
                test_ana2[abs(j[1] - j[0])] += 1

    ori_ana3_1 = {}
    pred_ana3_1 = {}
    ori_ana3_2 = {}
    pred_ana3_2 = {}
    ori_ana3_3 = {}
    pred_ana3_3 = {}

    for i in range(len(ori)):
        t = []
        for tt in test[i]:
            t += tt
        key = max(t) + 1
        if key not in ori_ana3_1.keys():
            ori_ana3_1[key] = len([l for l in ori[i] if l in test[i]])
        else:
            ori_ana3_1[key] += len([l for l in ori[i] if l in test[i]])

        if key not in ori_ana3_2.keys():
            ori_ana3_2[key] = len(ori[i])
        else:
            ori_ana3_2[key] += len(ori[i])

        if key not in ori_ana3_3.keys():
            ori_ana3_3[key] = len(test[i])
        else:
            ori_ana3_3[key] += len(test[i])

        if key not in pred_ana3_1.keys():
            pred_ana3_1[key] = len([l for l in pred[i] if l in test[i]])
        else:
            pred_ana3_1[key] += len([l for l in pred[i] if l in test[i]])

        if key not in pred_ana3_2.keys():
            pred_ana3_2[key] = len(pred[i])
        else:
            pred_ana3_2[key] += len(pred[i])

        if key not in pred_ana3_3.keys():
            pred_ana3_3[key] = len(test[i])
        else:
            pred_ana3_3[key] += len(test[i])


    ori_ana3_4 = {}
    pred_ana3_4 = {}
    for u in range(2, 38, 5):
    # for u in range(8,16,1):
        if u > 5:
            ori_uu = pred_uu = test_uu = 0
            for uu in range(u-5, u):
                if uu in ori_ana3_1.keys():
                    ori_uu += ori_ana3_1[uu]
                if uu in ori_ana3_2.keys():
                    pred_uu += ori_ana3_2[uu]
                if uu in ori_ana3_3.keys():
                    test_uu += ori_ana3_3[uu]
            ori_ana3_4[f"{u-5}-{u}"] = 2 * ori_uu / (pred_uu + test_uu)

            ori_uu = pred_uu = test_uu = 0
            for uu in range(u-5, u):
                if uu in pred_ana3_1.keys():
                    ori_uu += pred_ana3_1[uu]
                if uu in pred_ana3_2.keys():
                    pred_uu += pred_ana3_2[uu]
                if uu in pred_ana3_3.keys():
                    test_uu += pred_ana3_3[uu]
            pred_ana3_4[f"{u-5}-{u}"] = 2 * ori_uu / (pred_uu + test_uu)

    return ori_ana, pred_ana, ori_ana2, pred_ana2, ori_ana3_4, pred_ana3_4


def analysis2(ori, pred, test):
    ori_ana = {}
    pred_ana = {}
    for i in range(len(ori)):
        a1 = [l for l in ori[i] if l in test[i]]
        for j in a1:
            if abs(j[1] - j[0]) not in ori_ana.keys():
                ori_ana[abs(j[1] - j[0])] = 1
            else:
                ori_ana[abs(j[1] - j[0])] += 1

        a2 = [l for l in pred[i] if l in test[i]]
        for j in a2:
            if abs(j[1] - j[0]) not in pred_ana.keys():
                pred_ana[abs(j[1] - j[0])] = 1
            else:
                pred_ana[abs(j[1] - j[0])] += 1

    ori_ana2 = {}
    pred_ana2 = {}
    test_ana2 = {}
    for i in range(len(ori)):
        for j in ori[i]:
            if abs(j[1] - j[0]) not in ori_ana2.keys():
                ori_ana2[abs(j[1] - j[0])] = 1
            else:
                ori_ana2[abs(j[1] - j[0])] += 1

        for j in pred[i]:
            if abs(j[1] - j[0]) not in pred_ana2.keys():
                pred_ana2[abs(j[1] - j[0])] = 1
            else:
                pred_ana2[abs(j[1] - j[0])] += 1

        for j in test[i]:
            if abs(j[1] - j[0]) not in test_ana2.keys():
                test_ana2[abs(j[1] - j[0])] = 1
            else:
                test_ana2[abs(j[1] - j[0])] += 1

    ori_ana3_1 = {}
    pred_ana3_1 = {}
    ori_ana3_2 = {}
    pred_ana3_2 = {}
    ori_ana3_3 = {}
    pred_ana3_3 = {}

    for i in range(len(ori)):
        t = []
        for tt in test[i]:
            t += tt
        key = max(t) + 1
        if key not in ori_ana3_1.keys():
            ori_ana3_1[key] = len([l for l in ori[i] if l in test[i]])
        else:
            ori_ana3_1[key] += len([l for l in ori[i] if l in test[i]])

        if key not in ori_ana3_2.keys():
            ori_ana3_2[key] = len(ori[i])
        else:
            ori_ana3_2[key] += len(ori[i])

        if key not in ori_ana3_3.keys():
            ori_ana3_3[key] = len(test[i])
        else:
            ori_ana3_3[key] += len(test[i])

        if key not in pred_ana3_1.keys():
            pred_ana3_1[key] = len([l for l in pred[i] if l in test[i]])
        else:
            pred_ana3_1[key] += len([l for l in pred[i] if l in test[i]])

        if key not in pred_ana3_2.keys():
            pred_ana3_2[key] = len(pred[i])
        else:
            pred_ana3_2[key] += len(pred[i])

        if key not in pred_ana3_3.keys():
            pred_ana3_3[key] = len(test[i])
        else:
            pred_ana3_3[key] += len(test[i])


    ori_ana3_4 = {}
    pred_ana3_4 = {}
    # for u in range(2, 38, 5):
    for u in range(8,16,1):
        if u > 5:
            ori_uu = pred_uu = test_uu = 0
            for uu in range(u-1, u):
                if uu in ori_ana3_1.keys():
                    ori_uu += ori_ana3_1[uu]
                if uu in ori_ana3_2.keys():
                    pred_uu += ori_ana3_2[uu]
                if uu in ori_ana3_3.keys():
                    test_uu += ori_ana3_3[uu]
            ori_ana3_4[f"{u-1}-{u}"] = 2 * ori_uu / (pred_uu + test_uu)

            ori_uu = pred_uu = test_uu = 0
            for uu in range(u-1, u):
                if uu in pred_ana3_1.keys():
                    ori_uu += pred_ana3_1[uu]
                if uu in pred_ana3_2.keys():
                    pred_uu += pred_ana3_2[uu]
                if uu in pred_ana3_3.keys():
                    test_uu += pred_ana3_3[uu]
            pred_ana3_4[f"{u-1}-{u}"] = 2 * ori_uu / (pred_uu + test_uu)

    return ori_ana, pred_ana, ori_ana2, pred_ana2, ori_ana3_4, pred_ana3_4
