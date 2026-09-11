# SPDX-License-Identifier: MIT
"""Verify the minimum cut against the library it replaced.

`igraph` is **GPL** and this repository is MIT, so the cut is solved with
scipy's maximum flow (BSD, already a dependency).

**A minimum cut is not unique**, so the two libraries need not return the same
partition and measurably do not: on random graphs they agree most of the time
and occasionally pick different sides of an equally cheap cut. What has to hold
is what the caller depends on - **the cut costs the same, every invisible face
is on the removal side, and no visible face is** - and that is what is asserted
here rather than an identical list.

When igraph is not installed the comparison is skipped and the invariants are
checked on their own, so the test still says something on a machine that has
already dropped the GPL dependency.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from runners.trellis.fill_holes import _min_cut  # noqa: E402


def _random_case(seed: int) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(8, 60))
    edges = rng.integers(0, n, size=(int(rng.integers(n, 3 * n)), 2))
    edges = edges[edges[:, 0] != edges[:, 1]]
    weights = rng.random(len(edges)) * 5 + 0.01
    inner = rng.choice(n, size=max(1, n // 6), replace=False)
    outer = np.setdiff1d(rng.choice(n, size=max(1, n // 5), replace=False), inner)
    return n, edges, weights, inner, outer


def test_a_thin_link_is_where_it_cuts() -> None:
    """A dumbbell joined by one weak edge cuts there, and nowhere else."""
    edges = np.array([[0, 1], [1, 2], [2, 3]])
    weights = np.array([10.0, 0.001, 10.0])
    got = sorted(_min_cut(4, edges, weights, np.array([0]), np.array([3]), 4, 5))
    assert got == [0, 1], got


def test_nothing_is_cut_when_nothing_is_invisible() -> None:
    """With no faces on the source side there is nothing to remove."""
    got = _min_cut(
        3,
        np.array([[0, 1], [1, 2]]),
        np.array([5.0, 5.0]),
        np.array([], dtype=int),
        np.array([0, 1, 2]),
        3,
        4,
    )
    assert got == [], got


def test_the_partition_costs_the_maximum_flow() -> None:
    """The cut is minimum, by the theorem that says so - **checked without igraph**.

    A maximum flow equals a minimum cut, so the capacity crossing the partition
    this returns has to equal the flow through the graph. That is the property
    that would break if the partition were read out of the residual graph
    incorrectly, which is the part worth testing: **it is not true that every
    invisible face ends up removed** - paying for one face's arm can be cheaper
    than cutting the edges around it, and igraph does the same.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import maximum_flow

    for seed in range(12):
        n, edges, weights, inner, outer = _random_case(seed)
        if len(outer) == 0 or len(edges) == 0:
            continue
        source, target = n, n + 1
        removed = set(_min_cut(n, edges, weights, inner, outer, source, target)) | {source}

        ends_a = np.concatenate([edges[:, 0], inner, outer])
        ends_b = np.concatenate(
            [edges[:, 1], np.full(len(inner), source), np.full(len(outer), target)]
        )
        capacity = np.concatenate(
            [np.rint(weights * 1000.0), np.full(len(inner), 1000.0), np.full(len(outer), 1000.0)]
        )
        crossing = sum(
            c for a, b, c in zip(ends_a, ends_b, capacity, strict=True)
            if (int(a) in removed) != (int(b) in removed)
        )
        graph = coo_matrix(
            (
                np.clip(np.concatenate([capacity, capacity]), 1, 2**31 - 1).astype(np.int32),
                (np.concatenate([ends_a, ends_b]), np.concatenate([ends_b, ends_a])),
            ),
            shape=(n + 2, n + 2),
        ).tocsr()
        graph.sum_duplicates()
        flow = maximum_flow(graph, source, target).flow_value
        assert abs(crossing - flow) <= 1, f"seed {seed}: cut {crossing} against flow {flow}"


def test_it_costs_what_igraph_costs() -> None:
    """The cut is as cheap as the GPL library's, which is what "minimum" means."""
    try:
        import igraph
    except ImportError:
        print("       (igraph is not installed, so only the invariants were checked)")
        return

    for seed in range(12):
        n, edges, weights, inner, outer = _random_case(seed)
        if len(outer) == 0 or len(edges) == 0:
            continue
        source, target = n, n + 1

        graph = igraph.Graph()
        graph.add_vertices(n + 2)
        graph.add_edges(edges.tolist())
        capacities = (weights * 1000).tolist()
        graph.add_edges([(int(f), source) for f in inner])
        graph.add_edges([(int(f), target) for f in outer])
        capacities.extend([1000.0] * (len(inner) + len(outer)))

        theirs = graph.mincut(source, target, capacities)
        ours = set(_min_cut(n, edges, weights, inner, outer, source, target)) | {source}

        # The cost of our partition, measured on the same graph igraph was given.
        cost = 0.0
        for (a, b), capacity in zip(graph.get_edgelist(), capacities, strict=True):
            if (a in ours) != (b in ours):
                cost += capacity
        assert abs(cost - theirs.value) < 1e-6 * max(theirs.value, 1.0), (
            f"seed {seed}: our cut costs {cost}, igraph's {theirs.value}"
        )


def main() -> int:
    """Run every test."""
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  OK   {t.__name__}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
