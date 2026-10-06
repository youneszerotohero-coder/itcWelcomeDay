"""
sign_features.py — Shared feature extraction for the Shadow Clone sign
======================================================================
The Shadow Clone seal: index + middle fingers of BOTH hands extended and held
together, ring + pinky curled, and the two finger pairs crossing each other
(roughly a "+" shape) with the hands touching.

Raw screen coordinates depend on where the hands are in the frame, so instead
we describe the hands relative to each other (angles, finger extension, gap
between the fingers), normalised by hand size. Used by train.py and naruto.py.
"""

import numpy as np

ASPECT = 4 / 3  # camera frame width / height (640x480)

INDEX  = (5, 6, 7, 8)
MIDDLE = (9, 10, 11, 12)
RING   = (13, 14, 15, 16)
PINKY  = (17, 18, 19, 20)
SIGN_FINGERS = (6, 7, 8, 10, 11, 12)  # index + middle joints/tips

# Geometry gate thresholds (from the recorded shadow_clone samples)
MAX_FINGER_GAP = 0.35   # finger pairs must touch (in palm lengths)
MIN_CROSS_ANGLE = 20.0  # finger pairs must cross, not be parallel (degrees)
MIN_INDEX_EXT = 1.0     # index fingers must be extended
MAX_OPEN_EXT = 1.25     # a hand with ring AND pinky this extended is an open hand
# Where the two finger pairs cross, measured along each pair:
# 0 = knuckles, 1 = fingertips. Fingertips merely touching the other hand's
# fingers ("T" shape) lands at ~1.1+, a real cross lands inside both pairs.
CROSS_MIN = -0.25
CROSS_MAX = 1.0


def hands_from_raw(raw, aspect=ASPECT):
    """84 raw values (2 hands x 21 x (x, y)) -> list of (21, 2) arrays."""
    raw = np.asarray(raw, dtype=float)
    hands = []
    for k in range(2):
        p = raw[k * 42:(k + 1) * 42].reshape(21, 2).copy()
        if np.abs(p).sum() == 0:
            continue
        p[:, 0] *= aspect  # make x and y the same scale
        hands.append(p)
    return hands


def hands_from_result(result, aspect=ASPECT):
    """MediaPipe HandLandmarker result -> list of (21, 2) arrays."""
    hands = []
    for lms in (result.hand_landmarks or []):
        p = np.array([(lm.x * aspect, lm.y) for lm in lms], dtype=float)
        hands.append(p)
    return hands


def palm_size(p):
    return max(np.linalg.norm(p[9] - p[0]), 1e-6)


def extension(p, finger):
    """>1 means the finger is extended, <1 means curled."""
    mcp, pip, _, tip = finger
    return np.linalg.norm(p[tip] - p[0]) / max(np.linalg.norm(p[pip] - p[0]), 1e-6)


def finger_dir(p):
    """Unit direction of the index+middle finger pair."""
    d = (p[8] + p[12]) / 2 - (p[5] + p[9]) / 2
    return d / max(np.linalg.norm(d), 1e-6)


def finger_segment(p):
    """Centre line of the index+middle pair: knuckles -> fingertips."""
    return (p[5] + p[9]) / 2, (p[8] + p[12]) / 2


def crossing_point(h0, h1):
    """Where the two finger centre lines meet, as a position along each pair
    (0 = knuckles, 1 = fingertips). Clipped to [-3, 3]; parallel -> 3."""
    a0, a1 = finger_segment(h0)
    b0, b1 = finger_segment(h1)
    m = np.column_stack([a1 - a0, b0 - b1])
    if abs(np.linalg.det(m)) < 1e-9:
        return 3.0, 3.0
    s, t = np.linalg.solve(m, b0 - a0)
    return float(np.clip(s, -3, 3)), float(np.clip(t, -3, 3))


def geometry(h0, h1):
    size = (palm_size(h0) + palm_size(h1)) / 2
    cos = abs(np.clip(finger_dir(h0) @ finger_dir(h1), -1, 1))
    gap = min(np.linalg.norm(h0[a] - h1[b]) for a in SIGN_FINGERS for b in SIGN_FINGERS)
    c0, c1 = crossing_point(h0, h1)
    return {
        "angle": float(np.degrees(np.arccos(cos))),
        "finger_gap": float(gap / size),
        "wrist_dist": float(np.linalg.norm(h0[0] - h1[0]) / size),
        "min_index_ext": float(min(extension(h0, INDEX), extension(h1, INDEX))),
        "open_ext": float(max(min(extension(p, RING), extension(p, PINKY)) for p in (h0, h1))),
        "cross_0": c0,
        "cross_1": c1,
    }


def is_crossed(g):
    return (CROSS_MIN <= min(g["cross_0"], g["cross_1"])
            and max(g["cross_0"], g["cross_1"]) <= CROSS_MAX)


def passes_geometry(h0, h1):
    return geometry_ok(geometry(h0, h1))


def geometry_ok(g):
    return (g["finger_gap"] <= MAX_FINGER_GAP
            and g["angle"] >= MIN_CROSS_ANGLE
            and g["min_index_ext"] >= MIN_INDEX_EXT
            and g["open_ext"] <= MAX_OPEN_EXT
            and is_crossed(g))


def order_hands(h0, h1):
    """Put the more vertical finger pair first so hand order doesn't matter."""
    if abs(finger_dir(h0)[1]) >= abs(finger_dir(h1)[1]):
        return h0, h1
    return h1, h0


def features(h0, h1):
    a, b = order_hands(h0, h1)
    feats = []
    for p in (a, b):
        size = palm_size(p)
        feats += list(((p - p[0]) / size).ravel())               # hand shape (42)
        feats += [extension(p, f) for f in (INDEX, MIDDLE, RING, PINKY)]
        feats += list(finger_dir(p))
    g = geometry(a, b)
    size_a = palm_size(a)
    feats += [g["angle"] / 90, g["finger_gap"], g["wrist_dist"], g["min_index_ext"],
              g["cross_0"], g["cross_1"]]
    feats += list((b[0] - a[0]) / size_a)                        # where hand B sits
    mid_a = (a[8] + a[12] + a[6] + a[10]) / 4
    mid_b = (b[8] + b[12] + b[6] + b[10]) / 4
    feats += list((mid_b - mid_a) / size_a)                      # where the fingers meet
    feats.append(palm_size(b) / size_a)
    return np.array(feats, dtype=float)
