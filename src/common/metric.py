import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

def _scores_labels_from_block(Theta_hat, Theta_true, use_partial=False, tol=1e-12):
    """
    Extract scores and labels from a single (d x d) block (upper triangle only, excluding the diagonal).
    Score: |Theta_ij| or |partial corr|, Label: True non-zero (=1)/Zero (=0)
    """
    # symmetrize
    Th = 0.5 * (Theta_hat + Theta_hat.T)
    Tt = 0.5 * (Theta_true + Theta_true.T)
    d = Th.shape[0]
    iu = np.triu_indices(d, k=1)

    if use_partial:
        # -Theta_ij / sqrt(Theta_ii * Theta_jj)
        diag = np.clip(np.diag(Th), tol, None)
        denom = np.sqrt(diag[None, :] * diag[:, None])
        P = -Th / np.maximum(denom, tol)
        scores = np.abs(P[iu])
    else:
        scores = np.abs(Th[iu])

    labels = (np.abs(Tt[iu]) > tol).astype(int)
    return scores, labels

def ktvgl_auc(
    Thetas_hat, Thetas_true,
    metric="pr",            # "pr" (AUPRC) or "roc"
    aggregate="micro",      # "micro" or "macro"
    use_partial=False,
    tol=1e-12,
    skip_degenerate=True
):
    T = len(Thetas_hat)
    assert T == len(Thetas_true), "length mismatch on time axis"
    M = len(Thetas_hat[0])
    assert all(len(Thetas_hat[t]) == M for t in range(T)), "mode length mismatch"
    assert all(len(Thetas_true[t]) == M for t in range(T)), "mode length mismatch (true)"

    def auc_from(scores, labels):
        if (labels.sum() == 0) or (labels.sum() == labels.size):
            return np.nan
        if metric == "roc":
            return roc_auc_score(labels, scores)
        elif metric == "pr":
            return average_precision_score(labels, scores)
        else:
            raise ValueError("metric must be 'pr' or 'roc'")

    if aggregate == "micro":
        all_scores, all_labels = [], []
        for t in range(T):
            for m in range(M):
                s, y = _scores_labels_from_block(
                    Thetas_hat[t][m], Thetas_true[t][m],
                    use_partial=use_partial, tol=tol
                )
                all_scores.append(s)
                all_labels.append(y)
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        return auc_from(all_scores, all_labels)

    elif aggregate == "macro":
        aucs = []
        for t in range(T):
            for m in range(M):
                s, y = _scores_labels_from_block(
                    Thetas_hat[t][m], Thetas_true[t][m],
                    use_partial=use_partial, tol=tol
                )
                a = auc_from(s, y)
                if not np.isnan(a) or not skip_degenerate:
                    aucs.append(a)
        if len(aucs) == 0:
            return np.nan
        return float(np.nanmean(aucs))
    else:
        raise ValueError("aggregate must be 'micro' or 'macro'")


def ktvgl_eval(Thetas_hat, Thetas_true, aggregate="micro", use_partial=False, tol=1e-1):
    return {
        "ROC_AUC": ktvgl_auc(Thetas_hat, Thetas_true, metric="roc",
                             aggregate=aggregate, use_partial=use_partial, tol=tol),
        "AUPRC":   ktvgl_auc(Thetas_hat, Thetas_true, metric="pr",
                             aggregate=aggregate, use_partial=use_partial, tol=tol),
        "bestF1":  ktvgl_best_f1(Thetas_hat, Thetas_true, aggregate=aggregate,
                                 use_partial=use_partial, tol=tol),
    }


def _best_f1(scores, labels):
    pos = labels.sum()
    neg = labels.size - pos
    if pos == 0 or neg == 0:
        return 0.0, np.inf if pos == 0 else -np.inf, 0.0, 0.0

    precisions, recalls, thresholds = precision_recall_curve(labels, scores)
    P = precisions[:-1]
    R = recalls[:-1]
    F1 = (2 * P * R) / (P + R + 1e-12)
    idx = int(np.nanargmax(F1))
    return float(F1[idx]), float(thresholds[idx]), float(P[idx]), float(R[idx])


def ktvgl_best_f1(
    Thetas_hat, Thetas_true,
    aggregate="micro",   # "micro" or "macro"
    use_partial=False,
    tol=1e-12,
    skip_degenerate=True
):
    T = len(Thetas_hat)
    assert T == len(Thetas_true), "length mismatch on time axis"
    M = len(Thetas_hat[0])
    assert all(len(Thetas_hat[t]) == M for t in range(T))
    assert all(len(Thetas_true[t]) == M for t in range(T))

    if aggregate == "micro":
        all_scores, all_labels = [], []
        for t in range(T):
            for m in range(M):
                s, y = _scores_labels_from_block(
                    Thetas_hat[t][m], Thetas_true[t][m],
                    use_partial=use_partial, tol=tol
                )
                all_scores.append(s); all_labels.append(y)
        scores = np.concatenate(all_scores, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        f1, thr, prec, rec = _best_f1(scores, labels)
        return {"best_f1": f1, "threshold": thr, "precision": prec, "recall": rec}

    elif aggregate == "macro":
        per_block = []
        for t in range(T):
            for m in range(M):
                s, y = _scores_labels_from_block(
                    Thetas_hat[t][m], Thetas_true[t][m],
                    use_partial=use_partial, tol=tol
                )
                pos = y.sum()
                neg = y.size - pos
                if (pos == 0 or neg == 0) and skip_degenerate:
                    continue
                f1, thr, prec, rec = _best_f1(s, y)
                per_block.append({"t": t, "m": m, "best_f1": f1,
                                  "threshold": thr, "precision": prec, "recall": rec})
        if len(per_block) == 0:
            return {"mean_best_f1": np.nan, "per_block": per_block}
        mean_f1 = float(np.mean([d["best_f1"] for d in per_block]))
        return {"mean_best_f1": mean_f1, "per_block": per_block}

    else:
        raise ValueError("aggregate must be 'micro' or 'macro'")


def tvgl_auc(
    Thetas_hat_seq, Thetas_true_seq,
    metric="pr",          # "pr" (AUPRC) or "roc"
    aggregate="micro",    # "micro" or "macro"
    use_partial=False,
    tol=1e-12,
    skip_degenerate=True
):
    """
    Thetas_*_seq: np.ndarray 形状 (T, D, D)
    aggregate="micro": All time points combined into a single AUC
              "macro": Calculate the AUC for each time point and average them.
    """
    T = int(Thetas_hat_seq.shape[0])
    assert Thetas_true_seq.shape == Thetas_hat_seq.shape

    def auc_from(scores, labels):
        if labels.sum() == 0 or labels.sum() == labels.size:
            return np.nan
        if metric == "roc":
            return roc_auc_score(labels, scores)
        elif metric == "pr":
            return average_precision_score(labels, scores)
        else:
            raise ValueError("metric must be 'pr' or 'roc'")

    if aggregate == "micro":
        all_s, all_y = [], []
        for t in range(T):
            s, y = _scores_labels_from_block(Thetas_hat_seq[t], Thetas_true_seq[t],
                                             use_partial=use_partial, tol=tol)
            all_s.append(s); all_y.append(y)
        scores = np.concatenate(all_s, axis=0)
        labels = np.concatenate(all_y, axis=0)
        return float(auc_from(scores, labels))

    elif aggregate == "macro":
        aucs = []
        for t in range(T):
            s, y = _scores_labels_from_block(Thetas_hat_seq[t], Thetas_true_seq[t],
                                             use_partial=use_partial, tol=tol)
            a = auc_from(s, y)
            if not np.isnan(a) or not skip_degenerate:
                aucs.append(a)
        return float(np.nanmean(aucs)) if len(aucs) else np.nan
    else:
        raise ValueError("aggregate must be 'micro' or 'macro'")

def tvgl_best_f1(
    Thetas_hat_seq, Thetas_true_seq,
    aggregate="micro",    # "micro" or "macro"
    use_partial=False,
    tol=1e-12,
    skip_degenerate=True
):
    T = int(Thetas_hat_seq.shape[0])
    assert Thetas_true_seq.shape == Thetas_hat_seq.shape

    if aggregate == "micro":
        all_s, all_y = [], []
        for t in range(T):
            s, y = _scores_labels_from_block(Thetas_hat_seq[t], Thetas_true_seq[t],
                                             use_partial=use_partial, tol=tol)
            all_s.append(s); all_y.append(y)
        scores = np.concatenate(all_s, axis=0)
        labels = np.concatenate(all_y, axis=0)
        f1, thr, prec, rec = _best_f1(scores, labels)
        return {"best_f1": f1, "threshold": thr, "precision": prec, "recall": rec}

    elif aggregate == "macro":
        rows = []
        for t in range(T):
            s, y = _scores_labels_from_block(Thetas_hat_seq[t], Thetas_true_seq[t],
                                             use_partial=use_partial, tol=tol)
            pos = y.sum(); neg = y.size - pos
            if (pos == 0 or neg == 0) and skip_degenerate:
                continue
            f1, thr, prec, rec = _best_f1(s, y)
            rows.append({"t": t, "best_f1": f1, "threshold": thr,
                         "precision": prec, "recall": rec})
        mean_f1 = float(np.mean([r["best_f1"] for r in rows])) if rows else np.nan
        return {"mean_best_f1": mean_f1, "per_time": rows}
    else:
        raise ValueError("aggregate must be 'micro' or 'macro'")


# ====== wrapper ======
def tvgl_eval(
    Thetas_hat_seq, Thetas_true_seq,
    aggregate="micro", use_partial=False, tol=1e-1
):
    return {
        "ROC_AUC": tvgl_auc(Thetas_hat_seq, Thetas_true_seq,
                                      metric="roc", aggregate=aggregate,
                                      use_partial=use_partial, tol=tol),
        "AUPRC": tvgl_auc(Thetas_hat_seq, Thetas_true_seq,
                                    metric="pr", aggregate=aggregate,
                                    use_partial=use_partial, tol=tol),
        "bestF1": tvgl_best_f1(Thetas_hat_seq, Thetas_true_seq,
                                         aggregate=aggregate,
                                         use_partial=use_partial, tol=tol),
    }