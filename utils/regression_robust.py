import numpy as np

# Use the original optimal_point when it works, but provide robust fallbacks.
try:
    from utils.regression import optimal_point as _orig_optimal_point
except Exception:
    _orig_optimal_point = None


def optimal_point_robust(x, y, min_r2=0.8, target_count=10, start_degree=2, max_degree=20, min_improvement=0.003):
    """
    Robust optimal-point estimator.

    Strategy:
    1) Try the original polynomial-minima method (if available) with the same parameters.
    2) If it fails to produce a value, attempt knee/elbow detection via kneed.KneeLocator.
    3) Fall back to the discrete argmin of y.
    """
    # Step 1: original behavior
    if _orig_optimal_point is not None:
        try:
            return _orig_optimal_point(
                x,
                y,
                min_r2=min_r2,
                target_count=target_count,
                start_degree=start_degree,
                max_degree=max_degree,
                min_improvement=min_improvement,
            )
        except Exception:
            pass

    # Prepare arrays, sorted by x for stability
    xi = np.asarray(x, dtype=float)
    yi = np.asarray(y, dtype=float)
    order = np.argsort(xi)
    xi, yi = xi[order], yi[order]

    # Step 2: knee/elbow detection
    try:
        from kneed import KneeLocator  # type: ignore
        candidates = []
        for curve in ("convex", "concave"):
            for direction in ("increasing", "decreasing"):
                try:
                    kl = KneeLocator(xi, yi, curve=curve, direction=direction)
                    if kl.knee is not None:
                        candidates.append(float(kl.knee))
                except Exception:
                    continue
        if candidates:
            return float(np.median(candidates))
    except Exception:
        pass

    # Step 3: discrete argmin fallback
    return float(xi[np.argmin(yi)])

