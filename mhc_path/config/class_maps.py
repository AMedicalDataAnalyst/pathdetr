"""Dataset class maps and canonical taxonomy configuration.

Defines the canonical 5-class taxonomy for histopathology nuclei and provides
mappings from each supported dataset (CoNSeP, PanNuke, Lizard) to canonical
class indices. This module is the single source of truth for class semantics
across the entire mHC-Path pipeline.

The canonical ordering follows PanNuke convention, as established by the
CellViT and NuLite papers for cross-dataset comparability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

CANONICAL_CLASSES: tuple[str, ...] = (
    "neoplastic",   # 0
    "inflammatory",  # 1
    "epithelial",    # 2
    "connective",    # 3
    "dead",          # 4
)


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
    >>> cm = get_class_map("pannuke")
    >>> cm.remap_label(0, ("neoplastic", "inflammatory", "epithelial",
    ...                     "connective", "dead"))
    0
    """

    canonical_names: tuple[str, ...]
    source_to_canonical: dict[str, int]
    num_classes: int

    def remap_label(self, source_label: int, source_names: tuple[str, ...]) -> int:
        """Convert a source dataset integer label to the canonical class index.

        Parameters
        ----------
        source_label : int
            Integer label from the source dataset.
        source_names : tuple of str
            Ordered class names for the source dataset, so that
            ``source_names[source_label]`` gives the source class name.

        Returns
        -------
        int
            Canonical class index.

        Raises
        ------
        ValueError
            If ``source_label`` is out of range for ``source_names`` or the
            resulting source class name is not found in
            ``source_to_canonical``.

        Examples
        --------
        >>> cm = get_class_map("consep")
        >>> cm.remap_label(3, ("bg", "inflammatory", "epithelial",
        ...     "dysplastic", "fibroblast", "muscle", "endothelial"))
        0
        """
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


# ---------------------------------------------------------------------------
# Dataset-specific factory functions
# ---------------------------------------------------------------------------

def consep_class_map() -> ClassMap:
    """Create ClassMap for CoNSeP dataset.

    CoNSeP uses 1-indexed labels with the following semantics:

        1: inflammatory      -> canonical inflammatory (1)
        2: healthy epithelial -> canonical epithelial  (2)
        3: dysplastic/malignant epithelial -> canonical neoplastic  (0)
        4: fibroblast        -> canonical connective   (3)
        5: muscle            -> canonical connective   (3)
        6: endothelial       -> canonical connective   (3)

    Returns
    -------
    ClassMap
        Frozen mapping for CoNSeP.

    Examples
    --------
    >>> cm = consep_class_map()
    >>> cm.source_to_canonical["dysplastic"]
    0
    """
    # CoNSeP merges 3 stromal subtypes into connective, matching CellViT
    source_to_canonical = {
        "inflammatory": 1,
        "epithelial": 2,
        "dysplastic": 0,
        "fibroblast": 3,
        "muscle": 3,
        "endothelial": 3,
    }
    return ClassMap(
        canonical_names=CANONICAL_CLASSES,
        source_to_canonical=source_to_canonical,
        num_classes=len(CANONICAL_CLASSES),
    )


# CoNSeP source class names in label-index order (0 = background placeholder)
CONSEP_SOURCE_NAMES: tuple[str, ...] = (
    "bg",            # 0 — background, never used as a cell label
    "inflammatory",  # 1
    "epithelial",    # 2
    "dysplastic",    # 3
    "fibroblast",    # 4
    "muscle",        # 5
    "endothelial",   # 6
)


def pannuke_class_map() -> ClassMap:
    """Create ClassMap for PanNuke dataset.

    PanNuke raw .npy channel order differs from canonical at indices 2-4:

        raw ch0: neoplastic   -> canonical neoplastic   (0)
        raw ch1: inflammatory -> canonical inflammatory  (1)
        raw ch2: connective   -> canonical connective    (3)
        raw ch3: dead         -> canonical dead           (4)
        raw ch4: epithelial   -> canonical epithelial    (2)

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


# PanNuke raw .npy channel order (differs from canonical at indices 2-4)
PANNUKE_SOURCE_NAMES: tuple[str, ...] = (
    "neoplastic",    # channel 0
    "inflammatory",  # channel 1
    "connective",    # channel 2
    "dead",          # channel 3
    "epithelial",    # channel 4
)


def lizard_class_map() -> ClassMap:
    """Create ClassMap for Lizard dataset.

    Lizard uses 1-indexed labels. Four cell types merge into inflammatory:

        1: neutrophil  -> canonical inflammatory (1)
        2: epithelial  -> canonical epithelial   (2)
        3: lymphocyte  -> canonical inflammatory (1)
        4: plasma      -> canonical inflammatory (1)
        5: eosinophil  -> canonical inflammatory (1)
        6: connective  -> canonical connective   (3)

    Returns
    -------
    ClassMap
        Frozen mapping for Lizard.

    Examples
    --------
    >>> cm = lizard_class_map()
    >>> cm.source_to_canonical["lymphocyte"]
    1
    """
    # Lizard merges 4 immune subtypes into inflammatory, matching NuLite
    source_to_canonical = {
        "neutrophil": 1,
        "epithelial": 2,
        "lymphocyte": 1,
        "plasma": 1,
        "eosinophil": 1,
        "connective": 3,
    }
    return ClassMap(
        canonical_names=CANONICAL_CLASSES,
        source_to_canonical=source_to_canonical,
        num_classes=len(CANONICAL_CLASSES),
    )


# Lizard source class names in label-index order (0 = background placeholder)
LIZARD_SOURCE_NAMES: tuple[str, ...] = (
    "bg",          # 0 — background, never used as a cell label
    "neutrophil",  # 1
    "epithelial",  # 2
    "lymphocyte",  # 3
    "plasma",      # 4
    "eosinophil",  # 5
    "connective",  # 6
)


# ---------------------------------------------------------------------------
# Registry & factory
# ---------------------------------------------------------------------------

# New datasets: define a factory function above, then add one entry here.
_REGISTRY: dict[str, Callable[[], ClassMap]] = {
    "consep": consep_class_map,
    "pannuke": pannuke_class_map,
    "lizard": lizard_class_map,
}


def get_class_map(dataset_name: str) -> ClassMap:
    """Return the ClassMap for a supported dataset.

    Parameters
    ----------
    dataset_name : str
        Case-sensitive dataset identifier (e.g. ``"consep"``, ``"pannuke"``,
        ``"lizard"``).

    Returns
    -------
    ClassMap
        Frozen mapping for the requested dataset.

    Raises
    ------
    KeyError
        If ``dataset_name`` is not registered.

    Examples
    --------
    >>> cm = get_class_map("pannuke")
    >>> cm.num_classes
    5
    """
    if dataset_name not in _REGISTRY:
        raise KeyError(
            f"Unsupported dataset '{dataset_name}'. "
            f"Available: {sorted(_REGISTRY.keys())}"
        )
    return _REGISTRY[dataset_name]()
