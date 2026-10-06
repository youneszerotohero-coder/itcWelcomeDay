"""
train.py — Train Shadow Clone Sign Classifier
=============================================
Reads sign_data.csv and trains a gradient boosting model on hand-shape features
(see sign_features.py). Saves the model as sign_model.pkl

The recorded "other" data has very few two-hand wrong signs, so the model used
to learn "two hands = shadow clone". To fix that, we generate hard negatives
from the real shadow_clone samples:
  separated  — same hand shapes, but the hands are apart
  parallel   — finger pairs side by side instead of crossing
  open_hand  — ring + pinky extended on one/both hands
  fist       — index + middle curled on one hand
  mixed      — one correct hand next to a hand doing something else
  tip_touch  — fingertips only touch the other hand's fingers ("T", not "+")
and augment everything (rotation, scale, mirror, jitter).
"""

import csv
import pickle
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report

from sign_features import (hands_from_raw, features, geometry, passes_geometry,
                           palm_size, finger_dir, extension,
                           crossing_point, finger_segment,
                           INDEX, MIDDLE, RING, PINKY)

CSV_FILE   = "sign_data.csv"
MODEL_FILE = "sign_model.pkl"
THRESHOLD  = 0.6  # same confidence threshold naruto.py uses

rng = np.random.default_rng(42)


# ── Augmentation helpers ──
def rotate(p, center, deg):
    t = np.radians(deg)
    r = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    return (p - center) @ r.T + center


def finger_mid(p):
    return (p[6] + p[8] + p[10] + p[12]) / 4


def global_aug(h0, h1):
    """Same sign seen from a slightly different angle / distance / position."""
    center = np.vstack([h0, h1]).mean(0)
    deg, scale = rng.uniform(-35, 35), rng.uniform(0.6, 1.4)
    shift = rng.uniform(-0.3, 0.3, 2)
    noise = rng.uniform(0, 0.05)
    out = []
    for p in (h0, h1):
        q = rotate(p, center, deg)
        q = (q - center) * scale + center + shift
        q = q + rng.normal(0, noise * palm_size(q), q.shape)
        out.append(q)
    if rng.random() < 0.5:  # mirror (other hand on top)
        out = [np.column_stack([-q[:, 0], q[:, 1]]) for q in out]
    if rng.random() < 0.5:
        out = out[::-1]
    return out


def copy_finger(p, src, dst, scale=1.0):
    """Give finger `dst` the shape of finger `src` (anchored at dst's knuckle)."""
    p = p.copy()
    for s, d in zip(src, dst):
        p[d] = p[dst[0]] + (p[s] - p[src[0]]) * scale
    return p


def pos_variation(h0, h1):
    """Real sign held a bit differently: other cross angle, hands slid along
    each other. Only kept if the fingers still touch and cross."""
    for _ in range(10):
        b = rotate(h1, finger_mid(h1), rng.uniform(-20, 20))
        b = b + rng.uniform(-0.4, 0.4, 2) * palm_size(b)
        if passes_geometry(h0, b):
            return h0, b
    return h0, h1


def neg_separated(h0, h1):
    for _ in range(20):
        ang = rng.uniform(0, 2 * np.pi)
        d = np.array([np.cos(ang), np.sin(ang)]) * palm_size(h1) * rng.uniform(1.0, 4.0)
        h1b = h1 + d
        if geometry(h0, h1b)["finger_gap"] > 0.6:
            return h0, h1b
    return None


def neg_parallel(h0, h1):
    d0, d1 = finger_dir(h0), finger_dir(h1)
    target = np.degrees(np.arctan2(d0[1], d0[0]))
    if rng.random() < 0.5:
        target += 180
    cur = np.degrees(np.arctan2(d1[1], d1[0]))
    h1b = rotate(h1, finger_mid(h1), target - cur + rng.uniform(-12, 12))
    return h0, h1b


def neg_open_hand(h0, h1):
    hands = [h0, h1]
    for i in ([0], [1], [0, 1])[rng.integers(3)]:
        p = copy_finger(hands[i], MIDDLE, RING, rng.uniform(0.85, 1.0))
        hands[i] = copy_finger(p, MIDDLE, PINKY, rng.uniform(0.7, 0.85))
    return tuple(hands)


def neg_fist(h0, h1):
    hands = [h0, h1]
    i = rng.integers(2)
    if extension(hands[i], RING) > 1.0:  # ring not curled -> can't borrow its shape
        return None
    p = copy_finger(hands[i], RING, INDEX)
    hands[i] = copy_finger(p, RING, MIDDLE)
    return tuple(hands)


def neg_mixed(h0, h1, pool):
    other = pool[rng.integers(len(pool))]
    q = (other - other[0]) * (palm_size(h1) / palm_size(other))
    q = rotate(q, finger_mid(q), rng.uniform(0, 360))
    q = q - finger_mid(q) + finger_mid(h1) + rng.normal(0, 0.3 * palm_size(h1), 2)
    return (h0, q) if rng.random() < 0.5 else (q, h0)


def neg_tip_touch(h0, h1):
    """Slide one hand along its fingers so its fingertips just reach the
    other hand's fingers instead of crossing them."""
    hands = [h0, h1]
    i = rng.integers(2)
    c = crossing_point(h0, h1)[i]
    base, tip = finger_segment(hands[i])
    length = np.linalg.norm(tip - base)
    u = (tip - base) / length
    hands[i] = hands[i] + (c - rng.uniform(1.05, 1.5)) * length * u
    return tuple(hands)


def not_sign_hand(p):
    """A hand that clearly isn't the index+middle shape."""
    return (extension(p, RING) > 1.15 or extension(p, PINKY) > 1.15
            or extension(p, INDEX) < 0.95 or extension(p, MIDDLE) < 0.9)


# ── Load data ──
print("Loading data...")
pos, neg2, single = [], [], []
with open(CSV_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
        if not row:
            continue
        hands = hands_from_raw([float(v) for v in row[1:]])
        if row[0] == "shadow_clone" and len(hands) == 2:
            pos.append(hands)
        elif row[0] == "other" and len(hands) == 2:
            neg2.append(hands)
        elif row[0] == "other" and len(hands) == 1:
            single.append(hands[0])

uncrossed = [pair for pair in pos if not passes_geometry(*pair)]
pos = [pair for pair in pos if passes_geometry(*pair)]
print(f"shadow_clone (2 hands): {len(pos)}  "
      f"({len(uncrossed)} skipped: fingers touch but don't actually cross)")
print(f"other (2 hands):        {len(neg2)}")
print(f"other (1 hand):         {len(single)}  -> used to build mixed negatives")

flagged = [i for i, (a, b) in enumerate(neg2) if passes_geometry(a, b)]
if flagged:
    print(f"Note: {len(flagged)} 'other' sample(s) look like a crossed sign geometrically")


def build(pos_set, neg2_set, pool, n_pos=20, n_neg2=15, n_syn=3):
    X, y, src, gate = [], [], [], []

    def add(pair, label, source):
        if pair is None:
            return
        for _ in range(5):
            a, b = global_aug(*pair)
            # a "sign" sample must still be a real cross after augmenting
            if label == "other" or passes_geometry(a, b):
                X.append(features(a, b)); y.append(label); src.append(source)
                gate.append(passes_geometry(a, b))
                return

    for h0, h1 in pos_set:
        for _ in range(n_pos):
            add(pos_variation(h0, h1), "shadow_clone", "real_sign")
        for _ in range(n_syn):
            add(neg_separated(h0, h1), "other", "separated")
            add(neg_parallel(h0, h1), "other", "parallel")
            add(neg_open_hand(h0, h1), "other", "open_hand")
            add(neg_fist(h0, h1), "other", "fist")
            add(neg_mixed(h0, h1, pool), "other", "mixed")
            add(neg_tip_touch(h0, h1), "other", "tip_touch")
    for pair in neg2_set:
        for _ in range(n_neg2):
            add(pair, "other", "real_other")
    return np.array(X), np.array(y), np.array(src), np.array(gate)


def split(items, frac=0.2):
    idx = rng.permutation(len(items))
    k = max(1, int(len(items) * frac))
    return [items[i] for i in idx[k:]], [items[i] for i in idx[:k]]


def make_model():
    # predicts one frame in ~5 ms (a 150-tree random forest took ~20 ms)
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                          class_weight="balanced", random_state=42)


pool_all = [p for p in single + [h for pair in neg2 for h in pair] if not_sign_hand(p)]

# ── Evaluate on held-out recordings (split BEFORE augmenting, so no leakage) ──
pos_tr, pos_te = split(pos)
neg_tr, neg_te = split(neg2)
X_tr, y_tr, _, _ = build(pos_tr, neg_tr, pool_all)
X_te, y_te, src_te, gate_te = build(pos_te, neg_te, pool_all, n_pos=5, n_neg2=5, n_syn=2)

print(f"\nTraining on {len(y_tr)} samples (held-out test: {len(y_te)})...")
model = make_model()
model.fit(X_tr, y_tr)

proba = model.predict_proba(X_te)[:, list(model.classes_).index("shadow_clone")]
y_pred = np.where(proba >= THRESHOLD, "shadow_clone", "other")
print("\nHeld-out results:")
print(classification_report(y_te, y_pred))
app_pred = np.where(gate_te, y_pred, "other")  # naruto.py also needs the geometry gate
print("Per case, % correct (real_sign = detected, the rest = rejected):")
print("               model    app (model + geometry)")
for s in ["real_sign", "real_other", "separated", "parallel", "open_hand", "fist", "mixed", "tip_touch"]:
    m = src_te == s
    if m.any():
        acc = (y_pred[m] == y_te[m]).mean()
        app = (app_pred[m] == y_te[m]).mean()
        print(f"  {s:11s} {acc * 100:5.1f}%   {app * 100:5.1f}%   (n={m.sum()})")

# ── Final model on all data ──
X, y, _, _ = build(pos, neg2, pool_all)
print(f"\nTraining final model on all {len(y)} samples...")
model = make_model()
model.fit(X, y)

with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)

print(f"Model saved to {MODEL_FILE}")
