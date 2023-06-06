r"""Efficient kernel change point detection (dynamic programming)"""

from ruptures.base import BaseEstimator
from ruptures.costs import cost_factory
from ruptures.utils import from_path_matrix_to_bkps_list, sanity_check
from ruptures.exceptions import BadSegmentationParameters
import numpy as np

from ._detection.ekcpd import (
    ekcpd_cosine,
    ekcpd_Gaussian,
    ekcpd_L2,
    ekcpd_pelt_cosine,
    ekcpd_pelt_Gaussian,
    ekcpd_pelt_L2,
)


class KernelCPD(BaseEstimator):
    """Find optimal change points (using dynamic programming or pelt) for the
    special case where the cost function derives from a kernel function.

    Given a segment model, it computes the best partition for which the
    sum of errors is minimum.

    See the [user guide](../../../user-guide/detection/kernelcpd) for
    more information.
    """

    def __init__(self, kernel="linear", min_size=2, jump=1, params=None):
        r"""Creates a KernelCPD instance.

        Available kernels:

        - `linear`: $k(x,y) = x^T y$.
        - `rbf`: $k(x, y) = exp(\gamma \|x-y\|^2)$ where $\gamma>0$
        (`gamma`) is a user-defined parameter.
        - `cosine`: $k(x,y)= (x^T y)/(\|x\|\|y\|)$.

        Args:
            kernel (str, optional): name of the kernel, ["linear", "rbf", "cosine"]
            min_size (int, optional): minimum segment length.
            jump (int, optional): not considered, set to 1.
            params (dict, optional): a dictionary of parameters for the kernel instance

        Raises:
            AssertionError: if the kernel is not implemented.
        """
        self.kernel_name = kernel
        err_msg = "Kernel not found: {}.".format(self.kernel_name)
        assert self.kernel_name in ["linear", "rbf", "cosine"], err_msg
        self.model_name = "l2" if self.kernel_name == "linear" else self.kernel_name
        self.params = params
        # load the associated cost function
        if self.params is None:
            self.cost = cost_factory(model=self.model_name)
        else:
            self.cost = cost_factory(model=self.model_name, **self.params)
        self.min_size = max(min_size, self.cost.min_size)

        self.jump = 1  # set to 1
        self.n_samples = None
        self.segmentations_dict = dict()  # {n_bkps: bkps_list}

    def fit(self, signal) -> "KernelCPD":
        """Update some parameters (no computation in this function).

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        self.segmentations_dict = dict()
        self.cost.fit(signal.astype(np.double))
        self.n_samples = signal.shape[0]
        return self

    def predict(self, n_bkps=None, pen=None):
        """Return the optimal breakpoints. Must be called after the fit method.

        The breakpoints are associated with the signal passed to
        [`fit()`][ruptures.detection.kernelcpd.KernelCPD.fit].

        Args:
            n_bkps (int, optional): Number of change points. Defaults to None.
            pen (float, optional): penalty value (>0). Defaults to None. Not considered
                if n_bkps is not None.

        Raises:
            AssertionError: if `pen` or `n_bkps` is not strictly positive.
            BadSegmentationParameters: in case of impossible segmentation
                configuration

        Returns:
            list[int]: sorted list of breakpoints
        """
        # Our KernelCPD implementation with Pelt implies that we have at least one change point
        # raise an exception in case of impossible segmentation configuration
        if not sanity_check(
            n_samples=self.cost.signal.shape[0],
            n_bkps=1 if n_bkps is None else n_bkps,
            jump=self.jump,
            min_size=self.min_size,
        ):
            raise BadSegmentationParameters

        # dynamic programming if the user passed a number change points
        if n_bkps is not None:
            n_bkps = int(n_bkps)
            err_msg = "The number of changes must be positive: {}".format(n_bkps)
            assert n_bkps > 0, err_msg
            # if we have already computed it, return it without computations.
            if n_bkps in self.segmentations_dict:
                return self.segmentations_dict[n_bkps]
            # otherwise, call the C function
            if self.kernel_name == "linear":
                path_matrix_flat = ekcpd_L2(self.cost.signal, n_bkps, self.min_size)
            elif self.kernel_name == "rbf":
                path_matrix_flat = ekcpd_Gaussian(
                    self.cost.signal, n_bkps, self.min_size, self.cost.gamma
                )
            elif self.kernel_name == "cosine":
                path_matrix_flat = ekcpd_cosine(self.cost.signal, n_bkps, self.min_size)
            # from the path matrix, get all segmentation for k=1,...,n_bkps changes
            for k in range(1, n_bkps + 1):
                self.segmentations_dict[k] = from_path_matrix_to_bkps_list(
                    path_matrix_flat, k, self.n_samples, n_bkps, self.jump
                )
            return self.segmentations_dict[n_bkps]

        # Call pelt if the user passed a penalty
        if pen is not None:
            assert pen > 0, "The penalty must be positive: {}".format(pen)
            if self.kernel_name == "linear":
                path_matrix = ekcpd_pelt_L2(self.cost.signal, pen, self.min_size)
            elif self.kernel_name == "rbf":
                path_matrix = ekcpd_pelt_Gaussian(
                    self.cost.signal, pen, self.min_size, self.cost.gamma
                )
            elif self.kernel_name == "cosine":
                path_matrix = ekcpd_pelt_cosine(self.cost.signal, pen, self.min_size)

            my_bkps = list()
            ind = self.n_samples
            while ind > 0:
                my_bkps.append(ind)
                ind = path_matrix[ind]
            return my_bkps[::-1]

    def fit_predict(self, signal, n_bkps=None, pen=None):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            n_bkps (int, optional): Number of change points. Defaults to None.
            pen (float, optional): penalty value (>0). Defaults to None. Not considered if n_bkps is not None.

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(n_bkps=n_bkps, pen=pen)
