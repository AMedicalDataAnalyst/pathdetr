"""Canonical 5-class taxonomy for histopathology nuclei.

The canonical ordering follows PanNuke convention, as established by the
CellViT and NuLite papers for cross-dataset comparability.
"""

from __future__ import annotations

from dataclasses import dataclass

CANONICAL_CLASSES: tuple[str, ...] = (
    "neoplastic",   # 0
    "inflammatory",  # 1
    "epithelial",    # 2
    "connective",    # 3
    "dead",          # 4
)

NUM_CLASSES: int = len(CANONICAL_CLASSES)

# PanNuke source class names (identity mapping to canonical)
PANNUKE_SOURCE_NAMES: tuple[str, ...] = CANONICAL_CLASSES


@dataclass(frozen=True)
class ClassMap:
    """Immutable mapping from a source dataset's classes to canonical indices.

    Parameters
    ----------
    canonical_names : tuple of str
        The canonical class names in index order.
    source_to_canonical : dict of str to int
        Mapping from source class name to canonical class index.
    num_classes : int
        Number of canonical classes.

    Examples
    --------
    >>> cm = pannuke_class_map()
    >>> cm.remap_label(0, PANNUKE_SOURCE_NAMES)
    0
    """

    canonical_names: tuple[str, ...]
    source_to_canonical: dict[str, int]
    num_classes: int

    def remap_label(self, source_label: int, source_names: tuple[str, ...]) -> int:
        """Convert a source dataset integer label to the canonical class index."""
        if source_label < 0 or source_label >= len(source_names):
            raise ValueError(
                f"Source label {source_label} out of range for source_names "
                f"with {len(source_names)} entries."
            )
        source_name = source_names[source_label]
        if source_name not in self.source_to_canonical:
            raise ValueError(
                f"Unknown source class name '{source_name}' — not present in "
                f"source_to_canonical mapping. Known names: "
                f"{sorted(self.source_to_canonical.keys())}"
            )
        return self.source_to_canonical[source_name]


def pannuke_class_map() -> ClassMap:
    """Create ClassMap for PanNuke dataset (identity mapping).

    Returns
    -------
    ClassMap
        Frozen mapping for PanNuke.

    Examples
    --------
    >>> cm = pannuke_class_map()
    >>> cm.source_to_canonical["dead"]
    4
    """
    source_to_canonical = {name: i for i, name in enumerate(CANONICAL_CLASSES)}
    return ClassMap(
        canonical_names=CANONICAL_CLASSES,
        source_to_canonical=source_to_canonical,
        num_classes=len(CANONICAL_CLASSES),
    )


def get_class_map(dataset_name: str = "pannuke") -> ClassMap:
    """Return the ClassMap for PanNuke.

    Parameters
    ----------
    dataset_name : str
        Must be ``"pannuke"``.

    Returns
    -------
    ClassMap
        Frozen mapping for PanNuke.

    Examples
    --------
    >>> cm = get_class_map("pannuke")
    >>> cm.num_classes
    5
    """
    if dataset_name != "pannuke":
        raise KeyError(
            f"Unsupported dataset '{dataset_name}'. Only 'pannuke' is supported."
        )
    return pannuke_class_map()
