"""
clone_effect.py — Shadow Clone visual effect
============================================
1. MediaPipe Selfie Segmenter cuts the person out of each frame.
2. Copies of the person are pasted left/right, smaller and darker so they
   look further back (far ones first), then the real person goes on top.
3. Each clone replays a slightly older frame, so they follow you a beat late.
4. Smoke puffs (drawn in code) when the clones appear and vanish.

Timeline:  idle -> smoke_in -> clones -> smoke_out -> idle
"""

import collections
import time

import cv2
import mediapipe as mp
import numpy as np

SEGMENT_MODEL = "selfie_segmenter.tflite"

# (x offset as fraction of frame width, scale, brightness, frame delay)
# listed far -> near so nearer clones are drawn over farther ones
CLONES = [
    (-0.44, 0.70, 0.70, 12),
    ( 0.44, 0.70, 0.70,  9),
    (-0.25, 0.85, 0.85,  6),
    ( 0.25, 0.85, 0.85,  3),
]

SMOKE_TIME  = 0.45  # seconds for each puff of smoke
CLONE_TIME  = 5.0   # seconds the clones stay (doing the sign again extends it)
SMOKE_COLOR = (235, 235, 235)


def blend(top, bottom, alpha):
    """top * alpha + bottom * (1 - alpha), per pixel. uint8 images, float32
    alpha (h, w). OpenCV's blendLinear is far faster than numpy here."""
    return cv2.blendLinear(top, bottom, alpha, 1 - alpha)


def mask_box(mask):
    """(x, y, w, h) around the person; w == 0 if nobody was found."""
    return cv2.boundingRect((mask > 0.02).astype(np.uint8))


class CloneEffect:
    def __init__(self, model_path=SEGMENT_MODEL, clones=CLONES):
        options = mp.tasks.vision.ImageSegmenterOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        )
        self.segmenter = mp.tasks.vision.ImageSegmenter.create_from_options(options)
        self.clones    = clones
        self.history   = collections.deque(maxlen=max(c[3] for c in clones) + 1)
        self.state     = "idle"
        self.state_t   = 0.0
        self.particles = []  # [x, y, start radius, end radius, born, life]
        self.spawn_smoke = False
        self.last_ts   = -1
        self.rng       = np.random.default_rng()

    # ── Controls ──
    @property
    def active(self):
        return self.state != "idle" or bool(self.particles)

    def trigger(self, now=None):
        now = time.time() if now is None else now
        if self.state == "idle":
            self._set_state("smoke_in", now)
            self.history.clear()
            self.spawn_smoke = True
        elif self.state == "clones":
            self.state_t = now  # keep the clones longer

    def dismiss(self, now=None):
        now = time.time() if now is None else now
        if self.state in ("smoke_in", "clones"):
            self._set_state("smoke_out", now)
            self.spawn_smoke = True

    def close(self):
        self.segmenter.close()

    def _set_state(self, state, now):
        self.state, self.state_t = state, now

    # ── Per frame ──
    def apply(self, frame, rgb_small, now=None):
        """frame: BGR frame to draw on. rgb_small: same frame, RGB, any size
        (used for segmentation). Returns the frame with the effect."""
        now = time.time() if now is None else now
        if not self.active:
            return frame

        h, w = frame.shape[:2]
        elapsed = now - self.state_t
        if self.state == "smoke_in" and elapsed >= SMOKE_TIME:
            self._set_state("clones", now); elapsed = 0
        elif self.state == "clones" and elapsed >= CLONE_TIME:
            self._set_state("smoke_out", now); elapsed = 0
            self.spawn_smoke = True
        elif self.state == "smoke_out" and elapsed >= SMOKE_TIME:
            self._set_state("idle", now)

        # how visible the clones are right now
        if self.state == "smoke_in":
            visible = np.clip((elapsed / SMOKE_TIME - 0.4) / 0.6, 0, 1)
        elif self.state == "clones":
            visible = 1.0
        elif self.state == "smoke_out":
            visible = np.clip(1 - elapsed / (SMOKE_TIME * 0.5), 0, 1)
        else:
            visible = 0.0

        out = frame
        if self.state != "idle":
            mask = self._person_mask(rgb_small, w, h, now)
            box = mask_box(mask)
            self.history.append((frame.copy(), mask, box))
            if self.spawn_smoke:
                self._spawn_smoke(mask, w, h, now)
                self.spawn_smoke = False
            if visible > 0:
                out = self._draw_clones(frame, w, h, np.float32(visible))
                x, y, bw, bh = box  # the real you stays in front
                if bw:
                    out[y:y + bh, x:x + bw] = blend(frame[y:y + bh, x:x + bw],
                                                    out[y:y + bh, x:x + bw],
                                                    mask[y:y + bh, x:x + bw])

        return self._draw_smoke(out, w, h, now)

    # ── Pieces ──
    def _person_mask(self, rgb_small, w, h, now):
        ts = max(int(now * 1000), self.last_ts + 1)  # must strictly increase
        self.last_ts = ts
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(rgb_small))
        result = self.segmenter.segment_for_video(image, ts)
        conf = result.confidence_masks[0].numpy_view()
        conf = cv2.resize(np.squeeze(conf).astype(np.float32), (w, h))
        mask = np.clip((conf - 0.3) / 0.4, 0, 1)          # firmer edges
        return cv2.GaussianBlur(mask, (9, 9), 0).astype(np.float32)  # soft edges

    def _clone_transform(self, dx, scale, w, h):
        """Shrink towards the bottom centre (further away), shift sideways."""
        return np.float32([[scale, 0, (1 - scale) * w / 2 + dx * w],
                           [0, scale, (1 - scale) * h]])

    def _draw_clones(self, frame, w, h, visible):
        """Only the box around each clone is warped and blended, not the
        whole frame — several times faster."""
        out = frame.copy()
        for dx, scale, bright, delay in self.clones:
            idx = max(len(self.history) - 1 - delay, 0)
            src, src_mask, (bx, by, bw, bh) = self.history[idx]
            if not bw:
                continue
            m = self._clone_transform(dx, scale, w, h)
            x0 = max(int(m[0, 0] * bx + m[0, 2]), 0)
            y0 = max(int(m[1, 1] * by + m[1, 2]), 0)
            x1 = min(int(np.ceil(m[0, 0] * (bx + bw) + m[0, 2])) + 1, w)
            y1 = min(int(np.ceil(m[1, 1] * (by + bh) + m[1, 2])) + 1, h)
            if x0 >= x1 or y0 >= y1:
                continue  # clone is entirely off screen
            m[0, 2] -= x0
            m[1, 2] -= y0
            size = (x1 - x0, y1 - y0)
            img = cv2.warpAffine(src, m, size, flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT)
            a = cv2.warpAffine(src_mask, m, size, flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_CONSTANT) * visible
            img = cv2.convertScaleAbs(img, alpha=bright)  # darker = further back
            out[y0:y1, x0:x1] = blend(img, out[y0:y1, x0:x1], a)
        return out

    def _spawn_smoke(self, mask, w, h, now):
        ys, xs = np.nonzero(mask > 0.5)
        if len(xs) > 100:
            cx, cy = xs.mean(), ys.mean()
            spread_x, spread_y = xs.std(), ys.std()
        else:
            cx, cy, spread_x, spread_y = w / 2, h * 0.6, w * 0.1, h * 0.2
        for dx, scale, _, _ in self.clones:
            m = self._clone_transform(dx, scale, w, h)
            px, py = m @ np.array([cx, cy, 1.0])
            for _ in range(14):
                x = px + self.rng.normal(0, spread_x * scale * 0.6)
                y = py + self.rng.normal(0, spread_y * scale * 0.7)
                r0 = self.rng.uniform(10, 25) * scale
                r1 = self.rng.uniform(45, 80) * scale
                life = self.rng.uniform(0.5, 0.9)
                self.particles.append([x, y, r0, r1, now, life])

    def _draw_smoke(self, out, w, h, now):
        self.particles = [p for p in self.particles if now - p[4] < p[5]]
        if not self.particles:
            return out
        sw, sh = w // 2, h // 2  # half resolution: smoke is blurry anyway
        density = np.zeros((sh, sw), dtype=np.float32)
        for x, y, r0, r1, born, life in self.particles:
            t = (now - born) / life
            r = r0 + (r1 - r0) * (1 - (1 - t) ** 2)          # fast then slow
            strength = (1 - t) ** 1.5 * 0.8
            cx, cy, r = int(x / 2), int((y - 20 * t) / 2), int(r / 2)
            x0, y0 = max(cx - r, 0), max(cy - r, 0)
            x1, y1 = min(cx + r + 1, sw), min(cy + r + 1, sh)
            if x0 >= x1 or y0 >= y1:
                continue
            puff = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
            cv2.circle(puff, (cx - x0, cy - y0), r, strength, -1)
            region = density[y0:y1, x0:x1]
            np.maximum(region, puff, out=region)  # overlapping puffs blend
        density = np.clip(cv2.GaussianBlur(density, (0, 0), 6), 0, 0.9)
        density = cv2.resize(density, (w, h))
        smoke = np.empty_like(out)
        smoke[:] = SMOKE_COLOR
        return blend(smoke, out, density)
