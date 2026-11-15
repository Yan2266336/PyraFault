"""MINIMA's published LightGlue weights with the existing WSI adapter.

SuperPoint, MPP alignment, point matching settings, translation and reliability
are identical to our original LightGlue baseline. Only the learned LightGlue
checkpoint changes; this is a controlled WSI adaptation, not the author's demo.
"""
from __future__ import annotations

from pyrafault.verify.lightglue import LightGlueBaseline, OfficialSuperPointLightGlue


MINIMA_LIGHTGLUE_URL = (
    "https://github.com/LSXI7/storage/releases/download/MINIMA/minima_lightglue.pth"
)
# Measured from the public release linked by the author's MINIMA README.
MINIMA_LIGHTGLUE_SHA256 = (
    "bd981be2f5e8cdab120bb0380d132bf7d2ec20cd61499f92f283660bf2f5b126"
)


class OfficialMinimaLightGlue(OfficialSuperPointLightGlue):
    """Use MINIMA weights; keep the frozen, pinned native LightGlue backend."""

    def _expected_lightglue_sha256(self):
        return MINIMA_LIGHTGLUE_SHA256

    def __init__(self, source_root, superpoint_checkpoint, checkpoint, *, device="auto"):
        super().__init__(source_root, superpoint_checkpoint, checkpoint, device=device)
        self.configuration["checkpoint_family"] = "minima_lightglue"
        self.revision = f"minima-{self.revision}"


class MinimaLightGlueBaseline(LightGlueBaseline):
    """Same localisation contract and geometry, with a distinct method identity."""

    name = "minima_lightglue_baseline"

    def __init__(self, matcher, *, min_matches=4, translation_tolerance_px=3.0):
        super().__init__(matcher, min_matches=min_matches,
                         translation_tolerance_px=translation_tolerance_px)
        self.revision = f"minima-{self.revision}"
