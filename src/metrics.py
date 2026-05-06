import torch
from sklearn.metrics import f1_score
from utils import mat2bp
from sklearn.metrics import confusion_matrix

def _confusion_counts(ref, pred_binary):
    if ref.numel() == 0:
        return 0, 0, 0, 0
    tn, fp, fn, tp = confusion_matrix(ref.numpy(), pred_binary.numpy(), labels=[0, 1]).ravel()
    return int(tn), int(fp), int(fn), int(tp)


def binary_f1(ref_batch, pred_batch, zone_batch=None, th=0.5):
    """Compute F1 for 1D binary predictions. Input logits go through sigmoid then threshold."""
    f1_list = []
    tn_list, fp_list, fn_list, tp_list = [], [], [], []
    zone_tn_list, zone_fp_list, zone_fn_list, zone_tp_list = [], [], [], []
    non_zone_tn_list, non_zone_fp_list, non_zone_fn_list, non_zone_tp_list = [], [], [], []
    # Handle single sample case
    if len(ref_batch.shape) < 2:
        ref_batch = ref_batch.unsqueeze(0)
        pred_batch = pred_batch.unsqueeze(0)
        if zone_batch is not None and len(zone_batch.shape) < 2:
            zone_batch = zone_batch.unsqueeze(0)

    zone_iter = zone_batch if zone_batch is not None else [None] * len(ref_batch)
    for ref, pred, zone in zip(ref_batch, pred_batch, zone_iter):
        # Ignore padding (marked as -1)
        mask = ref != -1
        ref = ref[mask]
        pred = pred[mask]
        if zone is not None:
            zone = zone[mask]

        # Apply sigmoid and threshold
        pred = torch.sigmoid(pred)
        pred_binary = (pred > th).float()

        f1 = f1_score(ref.numpy(), pred_binary.numpy(), zero_division=0)
        tn, fp, fn, tp = _confusion_counts(ref, pred_binary)
        f1_list.append(f1)
        tn_list.append(tn)
        fp_list.append(fp)
        fn_list.append(fn)
        tp_list.append(tp)

        if zone is not None:
            zone_mask = zone == 1
            non_zone_mask = zone == 0

            zone_tn, zone_fp, zone_fn, zone_tp = _confusion_counts(ref[zone_mask], pred_binary[zone_mask])
            non_zone_tn, non_zone_fp, non_zone_fn, non_zone_tp = _confusion_counts(ref[non_zone_mask], pred_binary[non_zone_mask])

            zone_tn_list.append(zone_tn)
            zone_fp_list.append(zone_fp)
            zone_fn_list.append(zone_fn)
            zone_tp_list.append(zone_tp)
            non_zone_tn_list.append(non_zone_tn)
            non_zone_fp_list.append(non_zone_fp)
            non_zone_fn_list.append(non_zone_fn)
            non_zone_tp_list.append(non_zone_tp)

    out = {
        "f1": torch.tensor(f1_list).mean().item(),
        "tn": torch.tensor(tn_list).sum().item(),
        "fp": torch.tensor(fp_list).sum().item(),
        "fn": torch.tensor(fn_list).sum().item(),
        "tp": torch.tensor(tp_list).sum().item(),
        "zone_tn": 0,
        "zone_fp": 0,
        "zone_fn": 0,
        "zone_tp": 0,
        "non_zone_tn": 0,
        "non_zone_fp": 0,
        "non_zone_fn": 0,
        "non_zone_tp": 0,
    }
    if zone_batch is not None:
        out["zone_tn"] = torch.tensor(zone_tn_list).sum().item()
        out["zone_fp"] = torch.tensor(zone_fp_list).sum().item()
        out["zone_fn"] = torch.tensor(zone_fn_list).sum().item()
        out["zone_tp"] = torch.tensor(zone_tp_list).sum().item()
        out["non_zone_tn"] = torch.tensor(non_zone_tn_list).sum().item()
        out["non_zone_fp"] = torch.tensor(non_zone_fp_list).sum().item()
        out["non_zone_fn"] = torch.tensor(non_zone_fn_list).sum().item()
        out["non_zone_tp"] = torch.tensor(non_zone_tp_list).sum().item()
    return out


def contact_f1(ref_batch, pred_batch, Ls, th=0.5, reduce=True, method="triangular"):
    """Compute F1 from base pairs. Input goes to sigmoid and then thresholded"""
    f1_list = []

    if type(ref_batch) == float or len(ref_batch.shape) < 3:
        ref_batch = [ref_batch]
        pred_batch = [pred_batch]
        L = [L]

    for ref, pred, l in zip(ref_batch, pred_batch, Ls):
        # ignore padding
        ind = torch.where(ref != -1)
        pred = pred[ind].view(l, l)
        ref = ref[ind].view(l, l)

        # pred goes from -inf to inf
        pred = torch.sigmoid(pred)
        pred[pred<=th] = 0

        if method == "triangular":
            f1 = f1_triangular(ref, pred>0)
        if method == "f1_shift":
            ref_bp = mat2bp(ref)
            pred_bp = mat2bp(pred)
            f1 = f1_shift(ref_bp, pred_bp)
        
        f1_list.append(f1)

    if reduce:
        return torch.tensor(f1_list).mean().item()
    else:
        return torch.tensor(f1_list)


def f1_triangular(ref, pred):
    """Compute F1 from the upper triangular connection matrix"""
    # get upper triangular matrix without diagonal
    ind = torch.triu_indices(ref.shape[0], ref.shape[1], offset=1)

    ref = ref[ind[0], ind[1]].numpy().ravel()
    pred = pred[ind[0], ind[1]].numpy().ravel()

    return f1_score(ref, pred, zero_division=0)


def f1_strict(ref_bp, pre_bp):
    """F1 score strict, same as triangular but less efficient"""
    # corner case when there are no positives
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = 0
    for rbp in ref_bp:
        if rbp in pre_bp:
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if pbp in ref_bp:
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return tpr, pre, f1


def f1_shift(ref_bp, pre_bp):
    """F1 score with tolerance of 1 position"""
    # corner case when there are no positives
    if len(ref_bp) == 0 and len(pre_bp) == 0:
        return 1.0, 1.0, 1.0

    tp1 = 0
    for rbp in ref_bp:
        if (
            rbp in pre_bp
            or [rbp[0], rbp[1] - 1] in pre_bp
            or [rbp[0], rbp[1] + 1] in pre_bp
            or [rbp[0] + 1, rbp[1]] in pre_bp
            or [rbp[0] - 1, rbp[1]] in pre_bp
        ):
            tp1 = tp1 + 1
    tp2 = 0
    for pbp in pre_bp:
        if (
            pbp in ref_bp
            or [pbp[0], pbp[1] - 1] in ref_bp
            or [pbp[0], pbp[1] + 1] in ref_bp
            or [pbp[0] + 1, pbp[1]] in ref_bp
            or [pbp[0] - 1, pbp[1]] in ref_bp
        ):
            tp2 = tp2 + 1

    fn = len(ref_bp) - tp1
    fp = len(pre_bp) - tp1

    tpr = pre = f1 = 0.0
    if tp1 + fn > 0:
        tpr = tp1 / float(tp1 + fn)  # sensitivity (=recall =power)
    if tp1 + fp > 0:
        pre = tp2 / float(tp1 + fp)  # precision (=ppv)
    if tpr + pre > 0:
        f1 = 2 * pre * tpr / (pre + tpr)  # F1 score

    return tpr, pre, f1
