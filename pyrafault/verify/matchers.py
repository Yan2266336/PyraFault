"""Local correspondence between a downsampled query and a candidate window."""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

import cv2
import numpy as np

from pyrafault.geom import BBoxL0


@dataclass(frozen=True)
class MatchResult:
    """`score` is a composite: RANSAC inlier ratio times whichever of
    `geometric_consistency` / `structure_layout` the matcher was asked to use, so it
    stays a single number in [0, 1] no matter how many components are switched on --
    downstream code (ZoomTrace's argmax and decoy comparison) does not change shape.

    `geometric_consistency` and `structure_layout` are always populated with the real
    measured value when they can be computed (not gated to 1.0 just because a matcher
    was built with the component switched off), so a run can be re-scored under a
    different ablation setting from its own diagnostics without re-matching. They
    default to 1.0 -- neutral, i.e. "no penalty" -- only when there is nothing to
    measure (no transform, or fewer than 3 inliers), and for `MatchResult`s built
    directly (e.g. in tests) without setting them.
    """
    score: float                  # composite verification score, in [0, 1]
    n_matches: int
    n_inliers: int
    transform: np.ndarray | None  # 3x3, window -> query, or None
    geometric_consistency: float = 1.0  # linear-part-of-H rigidity, in [0, 1]; see below
    structure_layout: float = 1.0       # inlier spatial dispersion, in [0, 1]; see below


#: exp(-deviation / scale) falloff for `geometric_consistency_score`. Chosen from data,
#: not asserted from nothing: probing real LoFTR correspondences (see that function's
#: docstring) gave a maximum singular-value deviation of 0.0023 (median) at the true
#: location and 0.9124 at a random one. At scale=0.4 those map to exp(-0.0023/0.4) =
#: 0.994 and exp(-0.9124/0.4) = 0.102 -- the same separation the raw deviations show,
#: carried through onto a bounded [0, 1] scale instead of an unbounded one.
DEFAULT_GEOMETRIC_CONSISTENCY_SCALE = 0.4


def geometric_consistency_score(
    transform: np.ndarray | None, scale: float = DEFAULT_GEOMETRIC_CONSISTENCY_SCALE
) -> float:
    """How close the fitted transform's linear part is to a pure translation.

    Query and window are cropped at the same physical resolution (that is the whole
    point of `downsample_query` in zoomtrace.py), so a genuine correspondence should be
    explainable by translation alone: the 2x2 linear part `A` of the fitted homography
    `H` should be near the identity, not a spurious rotation/scale/shear that RANSAC
    happened to fit to noise. `A`'s two singular values are each 1 for a pure rotation
    (rotations preserve length) and pure translation contributes nothing to `A` at all,
    so `max(|singular_value - 1|)` is 0 exactly for translation/rotation and grows with
    any scale or shear component -- a single scalar "how non-rigid is this" measure.

    Mapped through an exponential falloff, `exp(-deviation / scale)`, rather than
    reported raw: raw deviation is unbounded above (an ill-conditioned `A` can have an
    arbitrarily large singular value), and the composite score this feeds into must
    stay in [0, 1] like the inlier ratio it multiplies. `scale` was chosen from real
    correspondences, not guessed -- see `DEFAULT_GEOMETRIC_CONSISTENCY_SCALE`.

    Returns 1.0 (neutral) when there is no transform to assess.
    """
    if transform is None:
        return 1.0
    linear = np.asarray(transform, dtype=np.float64)[:2, :2]
    singular_values = np.linalg.svd(linear, compute_uv=False)
    deviation = float(np.max(np.abs(singular_values - 1.0)))
    return float(np.exp(-deviation / scale))


def structure_layout_score(points: np.ndarray, side: int) -> float:
    """How much of the patch the inlier correspondences actually cover.

    The same number of inliers is stronger evidence spread across the patch than
    clustered in one corner: a cluster could be one repeated local texture agreeing by
    chance, while coverage of the whole patch is much harder to get from a spurious
    match. Coverage alone, though, is not the only shape "spread across the patch" can
    take: correspondences tracking a thin anatomical structure (a vessel, a duct, an
    edge) end to end are collinear or near-collinear by construction, and RANSAC still
    validated every one of them against the fitted homography. A convex hull has zero
    area for any collinear point set, so area alone cannot tell "a line running the
    length of the patch" apart from "four points piled on top of each other" -- yet
    those are opposite evidence. This function credits either shape of dispersion:

    - `coverage` = hull area / `side * side` (0 for < 3 points or a degenerate/collinear
      hull, same as before).
    - `extent` = the diagonal of the points' axis-aligned bounding box, over the patch's
      own diagonal `side * sqrt(2)` -- 1.0 exactly when the points reach from one patch
      corner to the opposite one, however thin the point set is otherwise. A line of
      points spanning the patch scores high here even though it scores 0 on `coverage`.

    `score = max(coverage, extent)`: both are independent pieces of evidence that the
    correspondences are NOT piled in one spot, so either being high is enough -- there
    is no reason a genuinely dispersed match should need both a hull with area AND a
    wide bounding box.

    A score of 1.0 means either the inliers' convex hull spans the entire side x side
    patch, or their bounding box reaches corner to corner (e.g. a line of points running
    the full diagonal or a full edge) -- both read as equally strong dispersal evidence.
    A score of 0.0 means fewer than 3 inlier points: a hull needs 3 points to enclose
    any area, and with only 1 or 2 points a bounding-box "extent" would just be the
    distance between two arbitrary points, not dispersion of a *matched set* -- there is
    no third point to confirm the first two aren't a fluke. This differs from the
    collinear case (3+ points, which now can score above 0 via `extent`): a homography
    RANSAC fit needs >= 4 points to begin with, so 3+ collinear inliers already carry
    more corroborating evidence than a bare pair ever can, even before asking whether
    they're spread out. Collinear points that are ALSO confined to a small region (not
    spanning the patch) still score low, exactly as before -- `extent` is small there
    too, so `coverage` being 0 costs nothing.

    `points` are Nx2 (x, y) inlier coordinates in the window's own pixel frame, N >= 0.
    """
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3 or side <= 0:
        return 0.0
    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull)
    coverage = float(np.clip(area / (side * side), 0.0, 1.0))

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    bbox_diagonal = float(np.linalg.norm(maxs - mins))
    patch_diagonal = side * math.sqrt(2.0)
    extent = float(np.clip(bbox_diagonal / patch_diagonal, 0.0, 1.0))

    return max(coverage, extent)


def _composite_score(inlier_ratio: float, geo: float, layout: float,
                     use_geo: bool, use_layout: bool) -> float:
    """Product of the inlier ratio and whichever components are switched on.

    A disabled component contributes a neutral 1.0 to the product -- it is not
    computed away, it is gated away, so the ablation that drops geometric consistency
    (or layout) reduces exactly to the inlier-ratio-only score this repo shipped with,
    rather than to some other combination.
    """
    return inlier_ratio * (geo if use_geo else 1.0) * (layout if use_layout else 1.0)


class Matcher(ABC):
    """Swap in LoFTR, RoMa or anything else without touching ZoomTrace."""

    name: str

    @abstractmethod
    def match(self, query: np.ndarray, window: np.ndarray) -> MatchResult:
        """Grayscale uint8 arrays of equal size in, correspondence summary out."""


class SiftRansacMatcher(Matcher):
    """SIFT keypoints, ratio-test matching, homography RANSAC.

    Kept as the dependency-free implementation of the interface. The base score is
    inliers / matches: the count alone is a proxy for overlapping area, not appearance.
    `use_geometric_consistency` and `use_structure_layout` fold in two more [0, 1]
    components as extra multiplicative factors (see `geometric_consistency_score` and
    `structure_layout_score`); both default OFF so the shipped score -- and the
    `thresholds.json` files already fitted against it -- are unchanged unless a caller
    opts in.
    """

    name = "sift_ransac"

    def __init__(self, ratio: float = 0.75, reproj_px: float = 3.0,
                 use_geometric_consistency: bool = False,
                 use_structure_layout: bool = False,
                 geometric_consistency_scale: float = DEFAULT_GEOMETRIC_CONSISTENCY_SCALE
                 ) -> None:
        self._sift = cv2.SIFT_create()
        self._ratio = ratio
        self._reproj = reproj_px
        self._use_geo = use_geometric_consistency
        self._use_layout = use_structure_layout
        self._geo_scale = geometric_consistency_scale

    def match(self, query: np.ndarray, window: np.ndarray) -> MatchResult:
        kq, dq = self._sift.detectAndCompute(query, None)
        kw, dw = self._sift.detectAndCompute(window, None)
        if dq is None or dw is None or len(kq) < 4 or len(kw) < 4:
            return MatchResult(0.0, 0, 0, None)

        pairs = cv2.BFMatcher().knnMatch(dw, dq, k=2)
        good = [a for a, b in (p for p in pairs if len(p) == 2)
                if a.distance < self._ratio * b.distance]
        if len(good) < 4:
            return MatchResult(0.0, len(good), 0, None)

        src = np.float32([kw[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kq[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, mask = cv2.findHomography(src, dst, cv2.RANSAC, self._reproj)
        if H is None or mask is None:
            return MatchResult(0.0, len(good), 0, None)

        n_in = int(mask.sum())
        inlier_ratio = n_in / len(good)
        geo = geometric_consistency_score(H, self._geo_scale)
        layout = structure_layout_score(src[mask.ravel() == 1], window.shape[0])
        composite = _composite_score(inlier_ratio, geo, layout,
                                     self._use_geo, self._use_layout)
        return MatchResult(composite, len(good), n_in, H,
                           geometric_consistency=geo, structure_layout=layout)


def refine_bbox(origin: tuple[int, int], side: int,
                transform: np.ndarray | None) -> BBoxL0 | None:
    """Where the query actually sits, from the correspondences rather than the window.

    Returning the window's own box would let grid alignment leak: a clean pair's true
    position lies on a lattice point while a drifted pair's necessarily falls between
    grid positions, so the drifted one would score worse for a geometric reason.
    Recovering a continuous position removes the asymmetry.
    """
    if transform is None:
        return None
    dx, dy = float(transform[0, 2]), float(transform[1, 2])
    # MINUS, not plus. findHomography(src=window_kp, dst=query_kp) returns H mapping
    # window coordinates to query coordinates, so the window-local point that lands on
    # the query's own origin is H^-1 . 0 = -(dx, dy), and its position in the context is
    # origin - (dx, dy). With the sign flipped the estimate is mirrored about the window
    # origin: measured against known ground truth it lands 22.4 px away where the
    # unrefined window box is 11.2 px away -- worse than not refining at all.
    return BBoxL0(int(round(origin[0] - dx)), int(round(origin[1] - dy)), side, side)


class LoftrMatcher(Matcher):
    """Detector-free dense matching via kornia's LoFTR.

    LoFTR is normally run at ~480 px and these windows are 65-129, so it was probed on
    real items before the design depended on it: at 128 px the true window gives 144
    matches against 2-24 for a random one, decaying to 18 at a full window's offset.
    The base score remains the RANSAC inlier ratio -- LoFTR's match count is the
    overlapping area, not the agreement. `use_geometric_consistency` and
    `use_structure_layout` fold in two more [0, 1] components as extra multiplicative
    factors (see `geometric_consistency_score` and `structure_layout_score`); both
    default OFF so the shipped score -- and the `thresholds.json` files already fitted
    against it -- are unchanged unless a caller opts in.
    """

    name = "loftr"

    def __init__(self, device: str | None = None, pretrained: str = "outdoor",
                 reproj_px: float = 3.0,
                 use_geometric_consistency: bool = False,
                 use_structure_layout: bool = False,
                 geometric_consistency_scale: float = DEFAULT_GEOMETRIC_CONSISTENCY_SCALE
                 ) -> None:
        import torch
        import kornia.feature as KF

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._net = KF.LoFTR(pretrained=pretrained).to(self._device).eval()
        self._reproj = reproj_px
        self._use_geo = use_geometric_consistency
        self._use_layout = use_structure_layout
        self._geo_scale = geometric_consistency_scale

    def _tensor(self, img: np.ndarray):
        a = np.asarray(img, dtype=np.float32) / 255.0
        return self._torch.from_numpy(a)[None, None].to(self._device)

    def match(self, query: np.ndarray, window: np.ndarray) -> MatchResult:
        with self._torch.no_grad():
            out = self._net({"image0": self._tensor(query),
                             "image1": self._tensor(window)})
        kq = out["keypoints0"].cpu().numpy()
        kw = out["keypoints1"].cpu().numpy()
        if len(kq) < 4:
            return MatchResult(0.0, len(kq), 0, None)

        H, mask = cv2.findHomography(
            kw.reshape(-1, 1, 2), kq.reshape(-1, 1, 2), cv2.RANSAC, self._reproj
        )
        if H is None or mask is None:
            return MatchResult(0.0, len(kq), 0, None)
        n_in = int(mask.sum())
        inlier_ratio = n_in / len(kq)
        geo = geometric_consistency_score(H, self._geo_scale)
        layout = structure_layout_score(kw[mask.ravel() == 1], window.shape[0])
        composite = _composite_score(inlier_ratio, geo, layout,
                                     self._use_geo, self._use_layout)
        return MatchResult(composite, len(kq), n_in, H,
                           geometric_consistency=geo, structure_layout=layout)


class RomaMatcher(Matcher):
    """Dense warp matching via RoMa (Edstedt et al.), as the third point of the matcher
    ablation beside LoFTR and SIFT.

    RoMa returns a dense correspondence field plus a per-pixel certainty rather than a
    keypoint list, so the obvious thing is to score it on its own terms -- mean
    certainty over the window, say. That is not done here. The ablation compares three
    matchers under one column header, and it only means something if the number is the
    same quantity in all three: the RANSAC inlier ratio, optionally multiplied by
    `geometric_consistency_score` and `structure_layout_score`. So the dense field is
    sampled down to correspondences and handed to exactly the path LoFTR's output takes.
    What differs between the three is the correspondences, which is the thing under test.

    Two practical notes. RoMa is trained at 560/864 px and these windows are 65-129, so
    it is fed upsampled -- the same accommodation LoFTR already gets, and the reason the
    ablation reports wall-clock beside accuracy. And it lives in an overlay environment
    (requirements-roma.txt), not the base one, so the import is deferred into `__init__`
    and the base environment stays as it was when the published numbers were produced.
    """

    name = "roma"

    #: How many correspondences to draw from the dense field. RoMa's own default is
    #: 10,000, which on a 129 px window is more points than pixels; 2,000 is ample for a
    #: homography and keeps the RANSAC cost in the same range as LoFTR's match counts.
    DEFAULT_SAMPLES = 2000

    #: RoMa is trained at 560 px coarse. A 65-129 px window upsampled to this gives the
    #: network something in its operating range; below ~256 it returns almost nothing.
    DEFAULT_SIDE = 560

    def __init__(self, device: str | None = None, variant: str = "outdoor",
                 reproj_px: float = 3.0, n_samples: int = DEFAULT_SAMPLES,
                 work_side: int = DEFAULT_SIDE,
                 use_geometric_consistency: bool = False,
                 use_structure_layout: bool = False,
                 geometric_consistency_scale: float = DEFAULT_GEOMETRIC_CONSISTENCY_SCALE
                 ) -> None:
        import torch
        try:
            import romatch
        except ImportError as exc:                       # pragma: no cover - env guard
            raise ImportError(
                "RoMa is not in this environment. It is an overlay: create it with "
                "`python -m venv --system-site-packages <path>` and "
                "`pip install -r requirements-roma.txt`. It is deliberately out of "
                "environment.lock.yml, which is the solve behind the published numbers."
            ) from exc

        self._torch = torch
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        build = romatch.roma_indoor if variant == "indoor" else romatch.roma_outdoor
        # `use_custom_corr=False`: the default path imports a separately compiled CUDA
        # extension named `local_corr` that pip does not build, and its absence surfaces
        # as `ModuleNotFoundError` from three frames inside the forward pass.
        self._net = build(device=self._device, use_custom_corr=False)
        self._reproj = reproj_px
        self._n_samples = n_samples
        self._side = work_side
        self._use_geo = use_geometric_consistency
        self._use_layout = use_structure_layout
        self._geo_scale = geometric_consistency_scale

    def _as_rgb(self, img: np.ndarray):
        """Grayscale uint8 -> the 3-channel PIL image RoMa expects, upsampled to its
        working resolution. Bilinear, matching how the window was produced.

        A PIL image and not a tensor: this version of `match` asserts
        `isinstance(im_A_input, Image.Image)` on the batched path, and the batched path
        is the only one left -- `batched=False` raises outright.
        """
        from PIL import Image
        a = np.asarray(img, dtype=np.uint8)
        return Image.fromarray(a, mode="L").convert("RGB").resize(
            (self._side, self._side), Image.BILINEAR)

    def match(self, query: np.ndarray, window: np.ndarray) -> MatchResult:
        side = int(np.asarray(window).shape[0])
        with self._torch.no_grad():
            # Batched is the only supported path, and it returns a leading batch axis
            # with the two directions concatenated along width: (1, H, 2W, 4) for the
            # warp and (1, H, 2W) for the certainty. `sample` wants one item, unbatched.
            warp, certainty = self._net.match(self._as_rgb(query), self._as_rgb(window),
                                              batched=True)
            if warp.dim() == 4:
                warp, certainty = warp[0], certainty[0]
            sampled, _ = self._net.sample(warp, certainty, num=self._n_samples)
            kq, kw = self._net.to_pixel_coordinates(
                sampled, self._side, self._side, self._side, self._side)
        kq = np.asarray(kq.cpu(), dtype=np.float32).reshape(-1, 2)
        kw = np.asarray(kw.cpu(), dtype=np.float32).reshape(-1, 2)
        # Back to the window's own pixel frame: everything downstream -- the homography,
        # `structure_layout_score`'s `side`, ZoomTrace's coordinate recovery -- is in it,
        # and leaving these at 560 would silently rescale the recovered transform.
        scale = side / float(self._side)
        kq, kw = kq * scale, kw * scale

        if len(kq) < 4:
            return MatchResult(0.0, len(kq), 0, None)
        H, mask = cv2.findHomography(
            kw.reshape(-1, 1, 2), kq.reshape(-1, 1, 2), cv2.RANSAC, self._reproj
        )
        if H is None or mask is None:
            return MatchResult(0.0, len(kq), 0, None)
        n_in = int(mask.sum())
        inlier_ratio = n_in / len(kq)
        geo = geometric_consistency_score(H, self._geo_scale)
        layout = structure_layout_score(kw[mask.ravel() == 1], side)
        composite = _composite_score(inlier_ratio, geo, layout,
                                     self._use_geo, self._use_layout)
        return MatchResult(composite, len(kq), n_in, H,
                           geometric_consistency=geo, structure_layout=layout)
