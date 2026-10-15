"""
models.py -- a small embedding-based ranking model written in plain NumPy
(so it runs anywhere, no PyTorch needed), plus training with distillation.

Model:   logit(user, item) = bias + user_bias + item_bias + <user_embedding, item_embedding>
         probability of click = sigmoid(logit)
"The large source model" and "the compact target model" are the SAME class with different
embedding sizes (16 vs 4 numbers per user/item).

TRAINING LOSS (the heart of distillation):
   loss = (1 - alpha) * BCE(real_label, p)  +  alpha * BCE(soft_label, p)
BCE's gradient w.r.t. the logit is (p - target), and the loss is linear in the target, so
this equals ordinary BCE against ONE blended target:  (1-alpha)*real + alpha*soft.
"""
import numpy as np
from data import sigmoid


class EmbeddingRanker:
    def __init__(self, n_users, n_items, dim, seed=0):
        r = np.random.default_rng(seed)
        self.dim = dim
        self.P = r.normal(0, 0.1, (n_users, dim))   # user embeddings
        self.Q = r.normal(0, 0.1, (n_items, dim))   # item (song) embeddings
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_items)
        self.b0 = 0.0

    def n_params(self):
        return self.P.size + self.Q.size + self.bu.size + self.bi.size + 1

    def logit(self, u, i):
        return (self.P[u] * self.Q[i]).sum(1) + self.bu[u] + self.bi[i] + self.b0

    def predict(self, u, i):
        return sigmoid(self.logit(u, i))

    def fit(self, u, i, y, soft=None, alpha=0.0, epochs=20, lr=0.02, l2=1e-4,
            batch=4096, seed=0, verbose=False):
        """Adam training. `soft` = source-model probabilities for the same pairs.
        alpha = 0 -> only real labels.  alpha = 1 -> only soft labels."""
        r = np.random.default_rng(seed)
        target = y if soft is None else (1 - alpha) * y + alpha * soft
        params = [self.P, self.Q, self.bu, self.bi]
        m = [np.zeros_like(p) for p in params] + [0.0]
        v = [np.zeros_like(p) for p in params] + [0.0]
        t = 0
        n = len(y)
        for ep in range(epochs):
            order = r.permutation(n)
            for s in range(0, n, batch):
                idx = order[s:s + batch]
                bu_, bi_, tg = u[idx], i[idx], target[idx]
                p = self.predict(bu_, bi_)
                g = (p - tg) / len(idx)                       # d loss / d logit
                gP = np.zeros_like(self.P); np.add.at(gP, bu_, g[:, None] * self.Q[bi_])
                gQ = np.zeros_like(self.Q); np.add.at(gQ, bi_, g[:, None] * self.P[bu_])
                gbu = np.zeros_like(self.bu); np.add.at(gbu, bu_, g)
                gbi = np.zeros_like(self.bi); np.add.at(gbi, bi_, g)
                gb0 = g.sum()
                gP += l2 * self.P; gQ += l2 * self.Q          # weight decay (stops over-fitting)
                grads = [gP, gQ, gbu, gbi, gb0]
                t += 1
                for k in range(5):                            # Adam update
                    m[k] = 0.9 * m[k] + 0.1 * grads[k]
                    v[k] = 0.999 * v[k] + 0.001 * grads[k] ** 2
                    step = lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
                    if k < 4:
                        params[k] -= step
                    else:
                        self.b0 -= step
            if verbose:
                print(f"  epoch {ep + 1}/{epochs}")
        return self

    # ---- "pick a few songs you like": build an embedding for a brand-new user ----
    def fold_in_user(self, liked, n_songs, steps=150, lr=0.05, l2=0.05, seed=0):
        """Item embeddings stay frozen; we learn only one new user vector from the liked songs
        (positives) and random other songs (assumed negatives). Returns scores for all songs."""
        r = np.random.default_rng(seed)
        liked = np.array(sorted(set(liked)))
        others = np.setdiff1d(np.arange(n_songs), liked)
        neg = r.choice(others, size=min(len(others), 8 * len(liked)), replace=False)
        items = np.concatenate([liked, neg])
        y = np.concatenate([np.ones(len(liked)), np.zeros(len(neg))])
        w = np.zeros(self.dim); b = 0.0
        mw = np.zeros(self.dim); vw = np.zeros(self.dim); mb = vb = 0.0
        for t in range(1, steps + 1):
            p = sigmoid(self.Q[items] @ w + self.bi[items] + b + self.b0)
            g = (p - y) / len(y)
            gw = self.Q[items].T @ g + l2 * w; gb = g.sum()
            mw = .9 * mw + .1 * gw; vw = .999 * vw + .001 * gw ** 2
            mb = .9 * mb + .1 * gb; vb = .999 * vb + .001 * gb ** 2
            w -= lr * (mw / (1 - .9 ** t)) / (np.sqrt(vw / (1 - .999 ** t)) + 1e-8)
            b -= lr * (mb / (1 - .9 ** t)) / (np.sqrt(vb / (1 - .999 ** t)) + 1e-8)
        return self.Q[:n_songs] @ w + self.bi[:n_songs] + b + self.b0

    def scores_for_user(self, u, n_songs):
        return self.logit(np.full(n_songs, u), np.arange(n_songs))
