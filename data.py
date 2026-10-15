"""
data.py  --  builds the synthetic "world" and the two interaction datasets.

IDEA OF THE SYNTHETIC WORLD
  * Every user has a hidden taste vector (how much they like each of 8 genres).
  * Every item has a hidden genre vector.
  * The chance that a user clicks an item depends on how well taste and genre match.
  * SOURCE domain = a big "video platform" log: all items (songs + other videos), lots of data.
  * TARGET domain = a small "music app" log: songs only, very little data.
  * The two domains are related but NOT identical (domain shift): in the music app
    users' tastes are slightly different and songs have their own popularity.
Nobody except this file ever sees the hidden vectors -- models only see (user, item, click).
"""
import numpy as np

GENRES = ["Pop", "Rock", "Hip-Hop", "Electronic", "Jazz", "Classical", "Indie", "Folk"]
_ADJ = ["Neon", "Golden", "Silent", "Electric", "Velvet", "Broken", "Midnight", "Crimson",
        "Hollow", "Lucky", "Paper", "Wild", "Lazy", "Burning", "Distant", "Frozen"]
_NOUN = ["Rain", "Highway", "Dreams", "Fire", "Echo", "Garden", "Summer", "Shadow",
         "River", "Heart", "City", "Skyline", "Ocean", "Moon", "Train", "Letters"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate(logits, target_rate):
    """Shift logits so the average click probability equals target_rate (bisection)."""
    lo, hi = -15.0, 15.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if sigmoid(logits + mid).mean() > target_rate:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def make_world(seed=0, n_users=1000, n_songs=300, n_other=1700,
               n_source=300_000, n_target_train=4_000, test_per_user=30):
    rng = np.random.default_rng(seed)
    n_items = n_songs + n_other           # items 0..n_songs-1 are songs, the rest are other videos
    K = len(GENRES)

    # ---- hidden user tastes: each user strongly likes 1-2 genres ----
    taste = rng.normal(0, 0.15, (n_users, K))
    for u in range(n_users):
        fav = rng.choice(K, size=rng.integers(1, 3), replace=False)
        taste[u, fav] += rng.uniform(0.7, 1.2, len(fav))
    # ---- hidden item genres: each item mostly belongs to one genre ----
    genre_of = rng.integers(0, K, n_items)
    item_vec = rng.normal(0, 0.15, (n_items, K))
    item_vec[np.arange(n_items), genre_of] += 1.0
    A = 4.0                                # how strongly taste-match drives clicks

    # ---- SOURCE-domain ground truth ----
    pop_src = rng.normal(0, 0.6, n_items)  # general item popularity
    act_src = rng.normal(0, 0.4, n_users)  # general user activity
    def src_logit(u, i):
        return A * np.sum(taste[u] * item_vec[i], 1) + pop_src[i] + act_src[u]

    # ---- TARGET-domain ground truth (related but shifted) ----
    taste_t = taste + rng.normal(0, 0.25, taste.shape)         # music-app taste differs a bit
    pop_tgt = 0.5 * pop_src + rng.normal(0, 0.6, n_items)      # songs have their own popularity
    act_tgt = 0.5 * act_src + rng.normal(0, 0.4, n_users)
    def tgt_logit(u, i):
        return A * np.sum(taste_t[u] * item_vec[i], 1) + pop_tgt[i] + act_tgt[u]

    # ---- sample SOURCE interactions (all items) ----
    su = rng.integers(0, n_users, n_source)
    si = rng.integers(0, n_items, n_source)
    lg = src_logit(su, si)
    lg = lg + _calibrate(lg, 0.15)
    sy = (rng.random(n_source) < sigmoid(lg)).astype(np.float32)

    # ---- sample TARGET train interactions (songs only) ----
    tu = rng.integers(0, n_users, n_target_train)
    ti = rng.integers(0, n_songs, n_target_train)
    # calibrate the offset on a large pilot sample so train/test share one click rate
    pu = rng.integers(0, n_users, 50_000); pi = rng.integers(0, n_songs, 50_000)
    off = _calibrate(tgt_logit(pu, pi), 0.25)
    ty = (rng.random(n_target_train) < sigmoid(tgt_logit(tu, ti) + off)).astype(np.float32)

    # ---- held-out TEST: every user gets `test_per_user` unseen candidate songs ----
    seen = [set() for _ in range(n_users)]
    for u, i in zip(tu, ti):
        seen[u].add(int(i))
    test_items = np.zeros((n_users, test_per_user), dtype=np.int64)
    for u in range(n_users):
        pool = np.array([s for s in range(n_songs) if s not in seen[u]])
        test_items[u] = rng.choice(pool, test_per_user, replace=False)
    test_users = np.repeat(np.arange(n_users), test_per_user).reshape(n_users, test_per_user)
    test_y = (rng.random(test_items.shape) <
              sigmoid(tgt_logit(test_users.ravel(), test_items.ravel()).reshape(test_items.shape) + off)
              ).astype(np.float32)

    # ---- readable song names ----
    names = [f"{_ADJ[rng.integers(16)]} {_NOUN[rng.integers(16)]} #{i}" for i in range(n_songs)]

    return dict(
        n_users=n_users, n_songs=n_songs, n_items=n_items,
        src_u=su, src_i=si, src_y=sy,
        tgt_u=tu, tgt_i=ti, tgt_y=ty,
        test_items=test_items, test_y=test_y,
        song_genre=genre_of[:n_songs], song_names=np.array(names),
        user_taste=taste_t,            # only used for DISPLAY in the app ("this user likes Jazz")
    )
