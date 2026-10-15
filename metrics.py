"""metrics.py -- AUC, precision@k, recall@k on the held-out test set."""
import numpy as np
from sklearn.metrics import roc_auc_score


def evaluate(model_predict, world, ks=(5, 10)):
    """model_predict(users, items)->probabilities. Test set = for each user, 30 unseen songs
    with real click labels. Models rank those 30 songs; we check how many top-k were clicked."""
    items, y = world["test_items"], world["test_y"]
    U, C = items.shape
    users = np.repeat(np.arange(U), C)
    s = model_predict(users, items.ravel()).reshape(U, C)
    out = {"AUC": roc_auc_score(y.ravel(), s.ravel())}
    order = np.argsort(-s, axis=1)
    ranked_y = np.take_along_axis(y, order, axis=1)
    total_pos = y.sum(1)
    ok = total_pos > 0                                   # users with at least one liked song
    for k in ks:
        hits = ranked_y[:, :k].sum(1)
        out[f"Precision@{k}"] = float((hits[ok] / k).mean())
        out[f"Recall@{k}"] = float((hits[ok] / total_pos[ok]).mean())
    return out
