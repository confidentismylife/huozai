"""Unqualified development probe of an intuitive smoke-only direction rule."""
from .timestamped_pipeline import CommonActionConstraints


class SmokePreferenceProbeConstraints(CommonActionConstraints):
    """Prefer the lower-smoke sign only when a lower-direction action is legal.

    This is an experimental comparison arm, not an adopted safety rule. It
    does not change physical corridor eligibility or override sign holds.
    """
    def action_mask(self, snapshot):
        mask = super().action_mask(snapshot)
        low = int(snapshot.smoke[4] < snapshot.smoke[3])
        high = 1 - low
        if (snapshot.smoke[3 + high] > snapshot.smoke[3 + low] + self.smoke_margin
                and mask[low * 4:low * 4 + 4].any()):
            mask[high * 4:high * 4 + 4] = False
        return mask
