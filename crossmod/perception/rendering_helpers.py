"""Shim: new home for rendering helpers.
Recommended import: from crossmod.perception.rendering_helpers import <name>
"""
try:
    from rendering_helpers import *  # legacy file
except Exception as e:
    raise ImportError('Expected legacy rendering_helpers.py; paste code here when fully migrated.') from e
