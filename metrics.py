import numpy as np

EPSILON = 5e-7


def _fast_hist(label_pred, label_true, num_classes):
    mask = (label_true >= 0) & (label_true < num_classes)
    hist = np.bincount(num_classes * label_true[mask].astype(int) + label_pred[mask],
                       minlength=num_classes ** 2).reshape(num_classes, num_classes)
    return hist


def eval_seg_metrics(predictions, gts, num_classes):
    hist = np.zeros((num_classes, num_classes))
    for lp, lt in zip(predictions, gts):
        hist += _fast_hist(lp.flatten(), lt.flatten(), num_classes)
    # Axis 0: gt, Axis 1: prediction
    acc = np.diag(hist).sum() / hist.sum()
    acc_cls = np.diag(hist) / hist.sum(axis=1)
    iou = (np.diag(hist)) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist) + EPSILON)
    mean_iou = np.nanmean(iou)
    freq = hist.sum(axis=1) / hist.sum()
    fwavacc = (freq[freq > 0] * iou[freq > 0].sum())
    return acc, acc_cls, mean_iou, fwavacc, iou
