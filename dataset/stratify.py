"""Stratification helpers for composite-eval reporting."""

from __future__ import annotations


def bucket_scope_size(n_scope: int) -> str:
    assert n_scope >= 0, "scope size cannot be negative"
    if n_scope <= 2:
        return "2"
    if n_scope <= 5:
        return "3-5"
    if n_scope <= 10:
        return "6-10"
    return ">10"


def bucket_composite_size(n_refs: int) -> str:
    assert n_refs >= 0, "composite size cannot be negative"
    if n_refs <= 4:
        return "small"
    if n_refs <= 10:
        return "medium"
    return "large"
