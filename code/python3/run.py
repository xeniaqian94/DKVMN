import numpy as np
import math
import mxnet as mx
import mxnet.ndarray as nd
from sklearn import metrics


def norm_clipping(params_grad, threshold):
    norm_val = 0.0
    for i in range(len(params_grad[0])):
        norm_val += np.sqrt(
            sum([nd.norm(grads[i]).asnumpy()[0] ** 2
                 for grads in params_grad]))
    norm_val /= float(len(params_grad[0]))

    if norm_val > threshold:
        ratio = threshold / float(norm_val)
        for grads in params_grad:
            for grad in grads:
                grad[:] *= ratio

    return norm_val


def binaryEntropy(target, pred, mod="avg"):
    loss = target * np.log(np.maximum(1e-10, pred)) + \
           (1.0 - target) * np.log(np.maximum(1e-10, 1.0 - pred))
    if mod == 'avg':
        return np.average(loss) * (-1.0)
    elif mod == 'sum':
        return - loss.sum()
    else:
        assert False


def compute_auc(all_target, all_pred):
    # fpr, tpr, thresholds = metrics.roc_curve(all_target, all_pred, pos_label=1.0)
    return metrics.roc_auc_score(all_target, all_pred)


def compute_f1(all_target, all_pred):
    return metrics.f1_score(all_target, all_pred)


def compute_accuracy(all_target, all_pred):
    all_pred[all_pred > 0.5] = 1.0
    all_pred[all_pred <= 0.5] = 0.0
    return metrics.accuracy_score(all_target, all_pred)


def train(net, params, q_data, qa_data, label):
    N = int(math.floor(len(q_data) / params.batch_size))
    print("train_N " + str(N))
    q_data = q_data.T  # Shape: (200,3633)
    qa_data = qa_data.T  # Shape: (200,3633)
    # Shuffle the data
    shuffled_ind = np.arange(q_data.shape[1])
    np.random.shuffle(shuffled_ind)
    q_data = q_data[:, shuffled_ind]
    qa_data = qa_data[:, shuffled_ind]

    pred_list = []
    target_list = []

    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    # init_memory_value = np.random.normal(0.0, params.init_std, ())
    for idx in range(N):
        if params.show: bar.next()

        q_one_seq = q_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        qa_one_seq = qa_data[:, idx * params.batch_size:(idx + 1) * params.batch_size]
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)

        target = qa_one_seq[:, :]
        # target = target.astype(np.int)
        # print(target)
        target = (target - 1) / params.n_question
        target = np.floor(target)
        # print(target)
        # target = target.astype(np.float) # correct: 1.0; wrong 0.0; padding -1.0

        input_q = mx.nd.array(input_q)
        input_qa = mx.nd.array(input_qa)
        target = mx.nd.array(target)

        data_batch = mx.io.DataBatch(data=[input_q, input_qa], label=[target])
        net.forward(data_batch, is_train=True)
        pred = net.get_outputs()[0].asnumpy()  # (seqlen * batch_size, 1)
        net.backward()

        norm_clipping(net._exec_group.grad_arrays, params.maxgradnorm)
        net.update()

        target = target.asnumpy().reshape((-1,))  # correct: 1.0; wrong 0.0; padding -1.0

        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist()
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    if params.show: bar.finish()

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    loss = binaryEntropy(all_target, all_pred)
    # print("all_target", all_target, len(all_target))
    # print("all_pred", all_pred, len(all_pred))
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc, compute_f1(all_target, all_pred)


def test(net, params, q_data, qa_data, label, split_data=None):
    # dataArray: [ array([[],[],..])] Shape: (3633, 200)
    N = int(math.ceil(float(len(q_data)) / float(params.batch_size)))
    q_data = q_data.T  # Shape: (200,3633) = seqlen*total_size
    qa_data = qa_data.T  # Shape: (200,3633)
    seq_num = q_data.shape[1]
    pred_list = []
    target_list = []
    if params.show:
        from utils import ProgressBar
        bar = ProgressBar(label, max=N)

    count = 0
    element_count = 0
    for idx in range(N):
        if params.show: bar.next()

        inds = np.arange(idx * params.batch_size, (idx + 1) * params.batch_size)
        q_one_seq = q_data.take(inds, axis=1, mode='wrap')
        qa_one_seq = qa_data.take(inds, axis=1, mode='wrap')
        # print 'seq_num', seq_num

        input_q = q_one_seq[:, :]  # Shape (seqlen, batch_size)
        input_qa = qa_one_seq[:, :]  # Shape (seqlen, batch_size)
        target = qa_one_seq[:, :]
        # target = target.astype(np.int)
        # target = (target - 1) / params.n_question
        # target = target.astype(np.float)  # correct: 1.0; wrong 0.0; padding -1.0
        target = (target - 1) / params.n_question
        target = np.floor(target)

        input_q = mx.nd.array(input_q)
        input_qa = mx.nd.array(input_qa)
        target = mx.nd.array(target)

        data_batch = mx.io.DataBatch(data=[input_q, input_qa], label=[])
        net.forward(data_batch, is_train=False)
        pred = net.get_outputs()[0].asnumpy()
        # print(pred.shape)
        target = target.asnumpy()
        if (idx + 1) * params.batch_size > seq_num:
            real_batch_size = seq_num - idx * params.batch_size
            target = target[:, :real_batch_size]
            pred = pred.reshape((params.seqlen, params.batch_size))[:,
                   :real_batch_size]  # numpy array of size seqlen*real_batch_size
            pred = pred.T.reshape((-1,))
            count += real_batch_size
        else:
            pred = pred.reshape((params.seqlen, params.batch_size)).T.reshape((-1,))
            count += params.batch_size

        target = target.T  # correct: 1.0; wrong 0.0; padding -1.0
        print("\ntarget/pred real shape "+str(target.shape)+" we then trim out padding per end of line wherever target=-1.0")  # we expect it to be batch_size * seqlen
        target=target.reshape((-1,))

        nopadding_index = np.flatnonzero(target != -1.0)
        nopadding_index = nopadding_index.tolist()
        # input(nopadding_index)
        pred_nopadding = pred[nopadding_index]
        target_nopadding = target[nopadding_index]

        element_count += pred_nopadding.shape[0]
        # print avg_loss
        pred_list.append(pred_nopadding)
        target_list.append(target_nopadding)

    if params.show: bar.finish()
    assert count == seq_num

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    print(all_pred, len(all_pred))
    print(all_target, len(all_target))

    if split_data is not None:
        # print(split_data)
        all_pred, all_target = trim_valid_only(all_pred, all_target, split_data)

    print("after trimming to have valid only...")
    print(all_pred, len(all_pred))
    print(all_target, len(all_target))
    print("Duolingo F1 is ", compute_duolingo_f1(all_target, all_pred))

    loss = binaryEntropy(all_target, all_pred)
    auc = compute_auc(all_target, all_pred)
    accuracy = compute_accuracy(all_target, all_pred)

    return loss, accuracy, auc, compute_f1(all_target, all_pred), compute_auroc(all_target, all_pred)


def trim_valid_only(all_pred, all_target, split_data):
    pred_list = []
    target_list = []

    start = 0
    end = 0
    for split_per_user in split_data:
        start = end + split_per_user[0]  # adds pure train length for this user
        end = start + split_per_user[1]  # adds pure valid length for this user
        pred_list.append(all_pred[start:end])
        target_list.append(all_target[start:end])

    all_pred = np.concatenate(pred_list, axis=0)
    all_target = np.concatenate(target_list, axis=0)

    return all_pred, all_target

def compute_auroc(actual, predicted):
    """
    Computes the area under the receiver-operator characteristic curve.
    This code a rewriting of code by Ben Hamner, available here:
    https://github.com/benhamner/Metrics/blob/master/Python/ml_metrics/auc.py
    """
    num = len(actual)
    temp = sorted([[predicted[i], actual[i]] for i in range(num)], reverse=True)

    sorted_predicted = [row[0] for row in temp]
    sorted_actual = [row[1] for row in temp]

    sorted_posterior = sorted(zip(sorted_predicted, range(len(sorted_predicted))))
    r = [0 for k in sorted_predicted]
    cur_val = sorted_posterior[0][0]
    last_rank = 0
    for i in range(len(sorted_posterior)):
        if cur_val != sorted_posterior[i][0]:
            cur_val = sorted_posterior[i][0]
            for j in range(last_rank, i):
                r[sorted_posterior[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_posterior)-1:
            for j in range(last_rank, i+1):
                r[sorted_posterior[j][1]] = float(last_rank+i+2)/2.0

    num_positive = len([0 for x in sorted_actual if x == 1])
    num_negative = num - num_positive
    sum_positive = sum([r[i] for i in range(len(r)) if sorted_actual[i] == 1])
    auroc = ((sum_positive - num_positive * (num_positive + 1) / 2.0) / (num_negative * num_positive))

    return auroc

def compute_duolingo_f1(actual, predicted):
    """
    Computes the F1 score of your predictions. Note that we use 0.5 as the cutoff here.
    """
    num = len(actual)

    true_positives = 0
    false_positives = 0
    false_negatives = 0
    true_negatives = 0

    for i in range(num):
        if actual[i] >= 0.5 and predicted[i] >= 0.5:
            true_positives += 1
        elif actual[i] < 0.5 and predicted[i] >= 0.5:
            false_positives += 1
        elif actual[i] >= 0.5 and predicted[i] < 0.5:
            false_negatives += 1
        else:
            true_negatives += 1

    try:
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        F1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        F1 = 0.0

    return F1

