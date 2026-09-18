# Algeon cuGraph fork

This repository is forked from NVIDIA [cuGraph](https://github.com/rapidsai/cugraph) at `v26.08.00`.

This fork keeps the upstream cuGraph codebase as its base and carries local changes needed by
`algeon-datafusion` (check [diff](https://github.com/algeon-dev/cugraph/compare/v26.08.00...algeon-dev:cugraph:algeon-26.08)).

## Fork Changes

The fork is intentionally limited to changes required by `algeon-datafusion`:

- Predicate-aware single-GPU breadth-first search across the C++, C, and Python APIs, including
  vertex and edge filters, target discovery, and device-resident edge-ID lowering.
- Host-side memory preflights for BFS and personalized PageRank side inputs and single-GPU graph
  construction, allowing downstream admission decisions before GPU allocation.
- Native integration hardening through typed allocation and invalid-vertex errors, explicit
  symmetric-graph rejection for strongly connected components, and reliable build-helper
  invocation.
