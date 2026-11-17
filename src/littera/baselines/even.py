from __future__ import annotations

from collections.abc import Sequence

from PIL import Image

from littera.baselines.common import placement_from_cuts
from littera.eval.ink import ink_profile
from littera.types import Placement, Point


class EvenPlacer:
    name = "even"

    def place(self, image: Image.Image, text: str, polygon: Sequence[Point] | None = None,
              baseline: Sequence[Point] | None = None) -> Placement:
        profile = ink_profile(image, polygon)
        left, right = profile.extent()
        units = max(1, len(text))
        step = (right - left) / units
        cuts = [left + step * i for i in range(len(text) + 1)]
        return placement_from_cuts(text, cuts, image.width, image.height, line_fit=0.5, top=profile.top, bottom=profile.bottom,
                                   placer=self.name)
