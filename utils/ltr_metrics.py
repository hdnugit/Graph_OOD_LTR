'''
This code is adapted from https://github.com/amazon-science/long-tailed-ood-detection/
'''
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
def accuracy_v2(pred_y, y):
    """Calculate accuracy."""
    return ((pred_y == y).sum() / len(y)).item()
    

def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out

def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95, pos_label=None):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                     np.array_equal(classes, [-1, 1]) or
                     np.array_equal(classes, [0]) or
                     np.array_equal(classes, [-1]) or
                     np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps      # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]

    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)      # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    return fps[cutoff] / (np.sum(np.logical_not(y_true)))   # , fps[cutoff]/(fps[cutoff] + tps[cutoff])

def get_measures(_pos, _neg, recall_level=0.95):
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = roc_auc_score(labels, examples)
    aupr = average_precision_score(labels, examples)
    fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)

    return auroc, aupr, fpr


def shot_acc(preds, labels, train_class_count, acc_per_cls=False):

    if isinstance(preds, torch.Tensor):
        preds = preds.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()
    elif isinstance(preds, np.ndarray):
        pass
    else:
        raise TypeError('Type ({}) of preds not supported'.format(type(preds)))

    num_classes = len(train_class_count)

    test_class_count = [np.nan] * num_classes
    class_correct = [np.nan] * num_classes
    for l in range(num_classes):
        test_class_count[l] = len(labels[labels == l])
        class_correct[l] = (preds[labels == l] == labels[labels == l]).sum()

    if num_classes <= 100: # e.g. On CIFAR10/100
        many_shot_thr = train_class_count[int(0.34*num_classes)]
        low_shot_thr = train_class_count[int(0.67*num_classes)]
    else:
        many_shot_thr=100
        low_shot_thr=20
    # print(many_shot_thr, low_shot_thr)

    many_shot = []
    median_shot = []
    low_shot = []
    for i in range(num_classes):
        if test_class_count[i] == 0:
            assert class_correct[i] == 0
            _acc_class_i = np.nan
        else:
            _acc_class_i = class_correct[i] / test_class_count[i]
        if train_class_count[i] > many_shot_thr:
            many_shot.append(_acc_class_i)
        elif train_class_count[i] < low_shot_thr:
            low_shot.append(_acc_class_i)
        else:
            median_shot.append(_acc_class_i)    

    # print('many_shot:', many_shot)
    # print('median_shot:', median_shot)
    # print('low_shot:', low_shot)

    if acc_per_cls:
        class_accs = [c / cnt for c, cnt in zip(class_correct, test_class_count)] 
        #print(class_correct, test_class_count, class_accs)
        return np.nanmean(many_shot), np.nanmean(median_shot), np.nanmean(low_shot), class_accs
    else:
        return np.nanmean(many_shot), np.nanmean(median_shot), np.nanmean(low_shot)
    