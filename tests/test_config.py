"""Tests for class_maps configuration."""

import pytest
from mhc_path.config.class_maps import (
    CANONICAL_CLASSES,
    NUM_CLASSES,
    PANNUKE_SOURCE_NAMES,
    ClassMap,
    get_class_map,
    pannuke_class_map,
)


def test_canonical_classes_count():
    assert NUM_CLASSES == 5


def test_canonical_class_names():
    assert CANONICAL_CLASSES == (
        "neoplastic", "inflammatory", "epithelial", "connective", "dead"
    )


def test_pannuke_identity_mapping():
    cm = pannuke_class_map()
    for i, name in enumerate(CANONICAL_CLASSES):
        assert cm.source_to_canonical[name] == i


def test_pannuke_remap_label():
    cm = pannuke_class_map()
    for i in range(NUM_CLASSES):
        assert cm.remap_label(i, PANNUKE_SOURCE_NAMES) == i


def test_pannuke_source_names_are_canonical():
    assert PANNUKE_SOURCE_NAMES == CANONICAL_CLASSES


def test_get_class_map_pannuke():
    cm = get_class_map("pannuke")
    assert isinstance(cm, ClassMap)
    assert cm.num_classes == 5


def test_get_class_map_unsupported():
    with pytest.raises(KeyError, match="Unsupported"):
        get_class_map("consep")


def test_remap_label_out_of_range():
    cm = pannuke_class_map()
    with pytest.raises(ValueError, match="out of range"):
        cm.remap_label(10, PANNUKE_SOURCE_NAMES)


def test_class_map_frozen():
    cm = pannuke_class_map()
    with pytest.raises(AttributeError):
        cm.num_classes = 10
