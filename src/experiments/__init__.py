"""Fleet scenario construction mechanism (recovered subset).

NOTE: This package intentionally uses a minimal initializer. The historical
``__init__.py`` on ``origin/cursor/campaign-clustering`` eagerly imports
``scenario_registry`` and ``result_writer``, which transitively depend on
orchestration modules that were never committed to git (see the prior forensic
audit / ``MISSING_MODULES.md``). To keep the recovered *construction mechanism*
importable in isolation — without inventing or substituting those missing
modules — only the byte-exact mechanism files are vendored here:

- ``data_splits.py``            (recovered byte-exact)
- ``vehicle_instance_builder.py`` (recovered byte-exact, then extended for
  opt-in frame-disjoint construction; historical default behavior preserved)
- ``frame_disjoint.py``         (new: source-frame overlap validator,
  per-run provenance, and construction checks)
"""
