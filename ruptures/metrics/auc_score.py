import numpy as np

from ruptures.utils import unzip


def auc(x, y):
    """Area under the curve using the composite trapezoidal rule.

    Assume that the x vary inside [0, 1].
    The curve is extended to fit inside [0, 1].

    Args:
        x (array): shape (n,)
        y (array): shape (n,)

    Returns:
        float: auc
    """
    points = list(zip(x, y))
    points.sort(key=lambda t: (t[0], -t[1]))
    # add extremities
    x0, y0 = points[0]
    if x0 > 0:
        points.insert(0, (0, y0))
    x0, y0 = points[-1]
    if x0 < 1:
        points.append((1, y0))
    xx, yy = unzip(points)
    area = np.trapz(yy, x=xx)
    return area
