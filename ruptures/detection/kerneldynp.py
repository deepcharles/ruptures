r"""Efficient kernel change point detection (dynamic programming)"""

from ruptures.utils import sanity_check
from ruptures.costs import cost_factory
from ruptures.base import BaseCost, BaseEstimator

from ruptures.detection._detection.ekcpd import ekcpd_L2, ekcpd_Gaussian
from ruptures.utils._utils.convert_path_matrix import from_path_matrix_to_bkps_list


class KernelDynp(BaseEstimator):

    """Find optimal change points (using dynamic programming) for the special
    case where the cost function derives from a kernel function.

    Given a segment model, it computes the best partition for which the
    sum of errors is minimum.
    """

    def __init__(self, kernel="linear", min_size=2, jump=5, params=None):
        r"""Creates a KernelDynp instance.

        Available kernels:

            - `linear`: $k(x,y) = x^T y$.
            - `rbf`: $k(x, y)$ = exp(\gamma \|x-y\|^2)$ where $\gamma>0$
            (`gamma`) is a user-defined parameter.

        Args:
            kernel (str, optional): name of the kernel, ["linear", "rbf"]
            min_size (int, optional): minimum segment length.
            jump (int, optional): subsample (one every *jump* points).
            params (dict, optional): a dictionary of parameters for the kernel instance
        """
        from_kernel_to_model_dict = {"linear": "l2", "rbf": "rbf"}
        self.kernel_name = kernel
        self.model_name = from_kernel_to_model_dict[kernel]
        self.params = params
        if self.params is None:
            self.cost = cost_factory(model=self.model_name)
        else:
            self.cost = cost_factory(model=self.model_name, **self.params)
        self.min_size = max(min_size, self.cost.min_size)
        self.jump = jump
        self.n_samples = None
        self.segmentations_dict = dict()  # {n_bkps: bkps_list}

    def fit(self, signal) -> "KernelDynp":
        """Update some parameters (no computation in this function).

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).

        Returns:
            self
        """
        # update some params
        self.segmentations_dict = dict()
        self.cost.fit(signal)
        self.n_samples = signal.shape[0]
        return self

    def predict(self, n_bkps):
        """Return the optimal breakpoints.

        Must be called after the fit method. The breakpoints are associated with the signal passed
        to [`fit()`][ruptures.detection.kerneldynp.KernelDynp.fit].

        Args:
            n_bkps (int): number of breakpoints.

        Returns:
            list: sorted list of breakpoints
        """
        n_bkps = int(n_bkps)
        if n_bkps in self.segmentations_dict:
            return self.segmentations_dict[n_bkps]

        if self.kernel_name == "linear":
            path_matrix_flat = ekcpd_L2(
                self.cost.signal, n_bkps, self.jump, self.min_size
            )
        elif self.kernel_name == "rbf":
            try:
                gamma = self.params["gamma"]
            except KeyError as e:
                msg = "Specify the parameter 'gamma' for the rbf kernel."
                raise e(msg)
            path_matrix_flat = ekcpd_Gaussian(
                self.cost.signal, n_bkps, self.jump, self.min_size, gamma
            )
        else:
            raise Exception("Kernel not found: {}.".format(self.kernel_name))

        for k in range(1, n_bkps + 1):
            self.segmentations_dict[k] = from_path_matrix_to_bkps_list(
                path_matrix_flat, k, self.n_samples, n_bkps, self.jump
            )

        return self.segmentations_dict[n_bkps]

    def fit_predict(self, signal, n_bkps):
        """Fit to the signal and return the optimal breakpoints.

        Helper method to call fit and predict once

        Args:
            signal (array): signal. Shape (n_samples, n_features) or (n_samples,).
            n_bkps (int): number of breakpoints.

        Returns:
            list: sorted list of breakpoints
        """
        self.fit(signal)
        return self.predict(n_bkps)
