import numpy as np


def calc_point2point(predict, actual):

    TP = np.sum(predict * actual)
    TN = np.sum((1 - predict) * (1 - actual))
    FP = np.sum(predict * (1 - actual))
    FN = np.sum((1 - predict) * actual)
    precision = TP / (TP + FP + 0.00001)
    recall = TP / (TP + FN + 0.00001)
    f1 = 2 * precision * recall / (precision + recall + 0.00001)
    return f1, precision, recall, TP, TN, FP, FN


def anomaly_interpretation(score, label, threhsold):
    if len(score) != len(label):
        raise ValueError("score and label must have the same length")

    score = np.asarray(score)
    label = np.asarray(label)

    predict = []

    for i in score:
        if i > threhsold:
            predict.append(1)
        else:
            predict.append(0)

    return predict * label


def adjust_predicts(score, label,
                    threshold=None,
                    pred=None,
                    calc_latency=False):

    if len(score) != len(label):
        raise ValueError("score and label must have the same length")
    score = np.asarray(score)
    label = np.asarray(label)
    latency = 0
    if pred is None:
        predict = score > threshold
    else:
        predict = pred
    actual = label > 0.1
    anomaly_state = False
    anomaly_count = 0
    for i in range(len(score)):
        if actual[i] and predict[i] and not anomaly_state:
                anomaly_state = True
                anomaly_count += 1
                for j in range(i, 0, -1):
                    if not actual[j]:
                        break
                    else:
                        if not predict[j]:
                            predict[j] = True
                            latency += 1
        elif not actual[i]:
            anomaly_state = False
        if anomaly_state:
            predict[i] = True
    if calc_latency:
        return predict, latency / (anomaly_count + 1e-4)
    else:
        return predict


def calc_seq(score, label, threshold, calc_latency=False):

    if calc_latency:
        predict, latency = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        t = list(calc_point2point(predict, label))
        t.append(latency)
        return t, predict
    else:
        predict = adjust_predicts(score, label, threshold, calc_latency=calc_latency)
        return calc_point2point(predict, label), predict


def bf_search(score, label):

    predict = 0.0
    m = (-1., -1., -1.)
    m_t = 0.0
    for i in range(98):
        threshold  = 0.01 * i + 0.01
        target, predict1 = calc_seq(score, label, threshold, calc_latency=True)
        if target[0] > m[0]:
            m_t = threshold
            m = target
            predict = predict1

    score = np.asarray(score)
    l1 = score>m_t
    point_label = l1 * label

    return m, m_t, predict, point_label


