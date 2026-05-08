"""Small helpers shared between the univariate and multivariate Pipelines."""
from __future__ import annotations

from infonet.core.base import Sensitivity


def flatten(args: tuple) -> list:
    """Flatten one level of nested lists/tuples in *args."""
    out = []
    for a in args:
        if isinstance(a, (list, tuple)):
            out.extend(a)
        else:
            out.append(a)
    return out


def sensitivity_combos(
    sensitivities: list[Sensitivity],
) -> list[list[Sensitivity]]:
    """Return one combo per sensitivity, or [[]] if none configured."""
    if not sensitivities:
        return [[]]
    return [[s] for s in sensitivities]


def collect_identity_defaults(sensitivities: list[Sensitivity]) -> dict:
    """Build a dict of `sens.<param>` keys to identity values."""
    out: dict = {}
    for s in sensitivities:
        for k, v in s.identity_params.items():
            out[f"sens.{k}"] = v
    return out
