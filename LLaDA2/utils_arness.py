from __future__ import annotations
from typing import Iterable, List, Dict, Sequence, Optional

def local_ar_ness(order: Sequence[int], max_k: int) -> Dict[int, float]:
    """
    Local AR-ness@k:
      fraction of steps t where (i_{t-k}, ..., i_t) forms a strictly increasing
      consecutive sequence: i_{j+1} = i_j + 1 for all j in [t-k, t-1].

    Notes:
      - k=1 is "next-token" pattern: i_t == i_{t-1} + 1.
      - Requires at least k+1 decoded tokens to evaluate a step.
    """
    if max_k < 1:
        raise ValueError("max_k must be >= 1")
    n = len(order)
    if n == 0:
        return {k: 0.0 for k in range(1, max_k + 1)}

    out: Dict[int, float] = {}
    for k in range(1, max_k + 1):
        if n < k + 1:
            out[k] = 0.0
            continue

        hits = 0
        total = n - k  # steps t = k..n-1 inclusive
        for t in range(k, n):
            ok = True
            # Check i_{t-k} -> ... -> i_t is consecutive increasing by 1 each step
            for j in range(t - k, t):
                if order[j + 1] != order[j] + 1:
                    ok = False
                    break
            if ok:
                hits += 1
        out[k] = hits / total

    return out


def global_ar_ness(order: Sequence[int], max_k: int, universe: Optional[Iterable[int]] = None) -> Dict[int, float]:
    """
    Global AR-ness@k:
      fraction of steps t where i_t is among the earliest k positions
      in the set of *remaining* (still-masked) positions.

    universe:
      - If None, we assume the universe is exactly the set of positions appearing in `order`.
      - If you want to include positions that were never decoded (unusual under your assumption),
        pass them explicitly via `universe`.

    Complexity:
      - O(N^2) in worst case due to maintaining remaining positions as a sorted list.
      - Fine for typical analysis sizes; can be optimized if needed.
    """
    if max_k < 1:
        raise ValueError("max_k must be >= 1")
    n = len(order)
    if n == 0:
        return {k: 0.0 for k in range(1, max_k + 1)}

    U = sorted(set(universe) if universe is not None else set(order))
    remaining = U[:]  # sorted list
    rem_set = set(remaining)

    # Basic sanity: order positions should be in universe
    for x in order:
        if x not in rem_set:
            raise ValueError(f"Order contains position {x} not in universe or duplicated unexpectedly.")

    hits = {k: 0 for k in range(1, max_k + 1)}
    total = 0

    for x in order:
        # earliest positions among remaining are remaining[0:k]
        # if k > len(remaining), "earliest k" means "all remaining"
        for k in range(1, max_k + 1):
            kk = min(k, len(remaining))
            if kk > 0 and x in remaining[:kk]:
                hits[k] += 1

        total += 1

        # remove x from remaining (monotone unmasking assumption)
        rem_set.remove(x)
        # list remove is O(N); ok for moderate N
        remaining.remove(x)

    return {k: hits[k] / total for k in range(1, max_k + 1)}


# ---------- quick example ----------
if __name__ == "__main__":
    order = [1, 2, 3, 4, 5, 6, 7]  # decoded positions per step
    print("\nOrder:", order)
    print("Local:", local_ar_ness(order, max_k=2))
    print("Global:", global_ar_ness(order, max_k=2))

    order = [5, 6, 7, 2, 3, 4, 1]  # decoded positions per step
    print("\nOrder:", order)
    print("Local:", local_ar_ness(order, max_k=2))
    print("Global:", global_ar_ness(order, max_k=2))

    order = [7, 6, 5, 4, 3, 2, 1]  # decoded positions per step
    print("\nOrder:", order)
    print("Local:", local_ar_ness(order, max_k=2))
    print("Global:", global_ar_ness(order, max_k=2))
