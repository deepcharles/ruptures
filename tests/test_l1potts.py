import subprocess
import sys
import warnings

import numpy as np
import pytest

import ruptures as rpt
from ruptures.costs import CostL1
from ruptures.detection import L1Potts, Pelt
from ruptures.exceptions import BadSegmentationParameters


@pytest.fixture(scope="module")
def signal_clean_step():
    return np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0, -2.0, -2.0])


def _pelt_l1_bkps(signal, pen):
    cost = CostL1()
    cost.min_size = 1
    return Pelt(custom_cost=cost, min_size=1, jump=1).fit(signal).predict(pen=pen)


def test_l1potts_top_level_export():
    assert rpt.L1Potts is L1Potts


def test_l1potts_clean_step(signal_clean_step):
    assert L1Potts().fit(signal_clean_step).predict(pen=0.5) == [3, 7, 9]


def test_l1potts_constant_signal():
    assert L1Potts().fit(np.zeros(20)).predict(pen=1.0) == [20]


def test_l1potts_single_sample():
    assert L1Potts().fit(np.array([3.14])).predict(pen=1.0) == [1]


def test_l1potts_shape_n_by_1():
    signal = np.array([[1.0], [1.0], [5.0], [5.0]])
    assert L1Potts().fit(signal).predict(pen=1.0) == [2, 4]


def test_l1potts_2d_rejected():
    with pytest.raises(BadSegmentationParameters):
        L1Potts().fit(np.zeros((10, 2)))


@pytest.mark.parametrize("seed", [1, 2, 3, 7, 42])
def test_l1potts_matches_pelt_functional(seed):
    sig, _ = rpt.pw_constant(
        n_samples=300, n_features=1, n_bkps=4, noise_std=0.3, seed=seed
    )
    pen = 0.4
    algo = L1Potts().fit(sig)
    f_l1 = algo._compute_functional_value(algo.predict(pen=pen), pen)
    f_pelt = algo._compute_functional_value(_pelt_l1_bkps(sig, pen), pen)
    assert f_l1 == pytest.approx(f_pelt, abs=1e-9)


def test_l1potts_weights_downweight_outlier():
    signal = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
    pen = 10.0
    bkps_uniform = L1Potts().fit(signal).predict(pen=pen)
    bkps_dw = (
        L1Potts()
        .fit(signal, weights=np.array([1.0, 1.0, 1e-3, 1.0, 1.0]))
        .predict(pen=pen)
    )
    assert bkps_uniform == [2, 3, 5]
    assert bkps_dw == [5]


@pytest.mark.parametrize(
    "signal,weights,pen",
    [
        (np.array([1.0, 5.0, 1.0]), np.array([2.0, 1.0, 3.0]), 0.7),
        (np.array([0.0, 3.0, 0.0, 7.0]), np.array([1.0, 4.0, 1.0, 2.0]), 1.5),
        (np.array([2.0, 2.0, 8.0, 2.0]), np.array([3.0, 1.0, 5.0, 2.0]), 2.0),
    ],
)
def test_l1potts_weights_match_replication(signal, weights, pen):
    """Integer weights are equivalent to sample replication.

    The optimal segmentation under weights w on signal f must match the
    optimal segmentation under uniform weights on np.repeat(f, w), after
    mapping breakpoints back through the cumulative-weight prefix.
    """
    bkps_direct = L1Potts().fit(signal, weights=weights).predict(pen=pen)

    int_w = weights.astype(int)
    rep_signal = np.repeat(signal, int_w)
    bkps_rep = L1Potts().fit(rep_signal).predict(pen=pen)

    cum = np.cumsum(int_w)
    n_rep = int(cum[-1])
    prefix_to_orig = {int(c): i + 1 for i, c in enumerate(cum)}
    expected = []
    for b in bkps_rep:
        if b == n_rep:
            expected.append(signal.shape[0])
        else:
            assert b in prefix_to_orig, "rep solution split inside an identical block"
            expected.append(prefix_to_orig[int(b)])
    assert bkps_direct == expected


@pytest.mark.parametrize(
    "signal,weights,pen",
    [
        # 100:1 ratio, alternating
        (
            np.array([0.0, 5.0, 0.0, 5.0, 0.0]),
            np.array([100.0, 1.0, 100.0, 1.0, 100.0]),
            0.5,
        ),
        # 1000:1 ratio
        (
            np.array([0.0, 5.0, 0.0, 5.0, 0.0]),
            np.array([1.0, 1000.0, 1.0, 1000.0, 1.0]),
            0.5,
        ),
        # one heavily weighted sample dominating, K = 2
        (
            np.array([0.0, 5.0, 5.0, 0.0]),
            np.array([1.0, 1.0, 10000.0, 1.0]),
            0.5,
        ),
        # extreme spread on a longer step signal
        (
            np.array([0.0, 0.0, 0.0, 5.0, 5.0, 5.0, -2.0, -2.0]),
            np.array([1.0, 50.0, 1.0, 1.0, 200.0, 1.0, 1.0, 30.0]),
            1.0,
        ),
        # outlier that would otherwise force a spurious segment, downweighted
        (
            np.array([0.0, 0.0, 100.0, 0.0, 0.0]),
            np.array([100.0, 100.0, 1.0, 100.0, 100.0]),
            0.5,
        ),
    ],
)
def test_l1potts_strongly_varying_weights_match_replication(signal, weights, pen):
    """Replication parity must hold even when weights span 2–4 orders of
    magnitude.

    Catches numerical drift, level-set bugs, or argmin tie-breaking
    issues that uniform-weight tests can't see.
    """
    bkps_direct = L1Potts().fit(signal, weights=weights).predict(pen=pen)

    int_w = weights.astype(int)
    rep_signal = np.repeat(signal, int_w)
    bkps_rep = L1Potts().fit(rep_signal).predict(pen=pen)

    cum = np.cumsum(int_w)
    n_rep = int(cum[-1])
    prefix_to_orig = {int(c): i + 1 for i, c in enumerate(cum)}
    expected = []
    for b in bkps_rep:
        if b == n_rep:
            expected.append(signal.shape[0])
        else:
            assert b in prefix_to_orig, "rep solution split inside an identical block"
            expected.append(prefix_to_orig[int(b)])
    assert bkps_direct == expected


@pytest.mark.parametrize("alpha", [1e-3, 1e-2, 0.5, 2.0, 100.0, 1e4])
def test_l1potts_weighted_homogeneity(alpha):
    """Scaling weights by alpha and pen by alpha must give the same bkps.

    The functional sum_i w_i |f_i - u_i| + pen * #jumps is positively
    homogeneous: scaling both terms by alpha doesn't change the
    minimizer. Verifies the weighted recurrence respects that invariance
    over a wide alpha range.
    """
    signal = np.array([0.0, 0.0, 5.0, 5.0, -2.0, -2.0, -2.0, 1.0, 1.0])
    weights = np.array([1.0, 2.0, 3.0, 1.0, 4.0, 1.0, 2.0, 5.0, 1.0])
    pen = 0.6
    base = L1Potts().fit(signal, weights=weights).predict(pen=pen)
    scaled = L1Potts().fit(signal, weights=alpha * weights).predict(pen=alpha * pen)
    assert base == scaled


def test_l1potts_zero_weight_sample_ignored():
    """A sample with weight 0 contributes nothing to the fit cost.

    With one outlier sample at weight 0, the optimum should match the
    optimum on the signal with that sample's value replaced by anything
    — in particular, by an adjacent value.
    """
    signal = np.array([0.0, 0.0, 999.0, 0.0, 0.0])
    weights = np.array([1.0, 1.0, 0.0, 1.0, 1.0])
    pen = 0.5
    bkps = L1Potts().fit(signal, weights=weights).predict(pen=pen)
    # The 999 outlier carries no weight, so the optimum should be one segment.
    assert bkps == [5]


def test_l1potts_dynamic_range_2k_signal():
    """Longer signal with non-trivial weight pattern: parity vs PELT-via-
    replication on a 200-sample signal with weights in {1, 5, 25}."""
    rng = np.random.default_rng(seed=7)
    n = 200
    sig, _ = rpt.pw_constant(n_samples=n, n_features=1, n_bkps=5, noise_std=0.4, seed=7)
    weights = rng.choice([1.0, 5.0, 25.0], size=n)
    pen = 1.0

    bkps_direct = L1Potts().fit(sig, weights=weights).predict(pen=pen)

    int_w = weights.astype(int)
    rep_signal = np.repeat(sig, int_w)
    bkps_rep = L1Potts().fit(rep_signal).predict(pen=pen)

    cum = np.cumsum(int_w)
    n_rep = int(cum[-1])
    prefix_to_orig = {int(c): i + 1 for i, c in enumerate(cum)}
    expected = []
    for b in bkps_rep:
        if b == n_rep:
            expected.append(n)
        else:
            assert b in prefix_to_orig, "rep solution split inside an identical block"
            expected.append(prefix_to_orig[int(b)])
    assert bkps_direct == expected


def test_l1potts_weighted_functional_matches_replication():
    """_compute_functional_value with weights == _compute_functional_value on
    the replicated unweighted signal."""
    signal = np.array([1.0, 5.0, 1.0])
    weights = np.array([3.0, 1.0, 2.0])
    pen = 0.7
    algo = L1Potts().fit(signal, weights=weights)
    bkps_w = algo.predict(pen=pen)
    fval_w = algo._compute_functional_value(bkps_w, pen)

    rep_signal = np.repeat(signal, weights.astype(int))
    algo_rep = L1Potts().fit(rep_signal)
    bkps_rep = algo_rep.predict(pen=pen)
    fval_rep = algo_rep._compute_functional_value(bkps_rep, pen)

    assert fval_w == pytest.approx(fval_rep)


def test_l1potts_idempotent(signal_clean_step):
    algo = L1Potts().fit(signal_clean_step)
    assert algo.predict(pen=0.5) == algo.predict(pen=0.5)


def test_l1potts_multi_pen_independence(signal_clean_step):
    """Fit() once, predict() with different penalties — results must not bleed
    between calls."""
    algo = L1Potts().fit(signal_clean_step)
    b_low = algo.predict(pen=0.1)
    b_high = algo.predict(pen=1000.0)
    b_low_again = algo.predict(pen=0.1)
    b_high_again = algo.predict(pen=1000.0)
    assert b_low == b_low_again
    assert b_high == b_high_again
    # very high penalty collapses to a single segment
    assert b_high == [signal_clean_step.shape[0]]
    assert b_low != b_high


def test_l1potts_reversal_same_functional():
    """Reversing the signal must yield a segmentation with the same functional
    value."""
    sig, _ = rpt.pw_constant(
        n_samples=200, n_features=1, n_bkps=4, noise_std=0.3, seed=11
    )
    pen = 0.4
    algo_fwd = L1Potts().fit(sig)
    algo_rev = L1Potts().fit(sig[::-1])
    f_fwd = algo_fwd._compute_functional_value(algo_fwd.predict(pen), pen)
    f_rev = algo_rev._compute_functional_value(algo_rev.predict(pen), pen)
    assert f_fwd == pytest.approx(f_rev, abs=1e-9)


def test_l1potts_fit_predict(signal_clean_step):
    assert L1Potts().fit_predict(signal_clean_step, pen=0.5) == [3, 7, 9]


def test_l1potts_fit_predict_passes_weights():
    signal = np.array([1.0, 1.0, 100.0, 1.0, 1.0])
    bkps = L1Potts().fit_predict(
        signal, pen=10.0, weights=np.array([1.0, 1.0, 1e-3, 1.0, 1.0])
    )
    assert bkps == [5]


def test_l1potts_invalid_pen():
    algo = L1Potts().fit(np.array([0.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        algo.predict(pen=0.0)
    with pytest.raises(ValueError):
        algo.predict(pen=-1.0)


def test_l1potts_pen_nan():
    algo = L1Potts().fit(np.array([0.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        algo.predict(pen=float("nan"))


def test_l1potts_invalid_weights_shape():
    with pytest.raises(ValueError):
        L1Potts().fit(np.array([0.0, 0.0, 1.0, 1.0]), weights=np.ones(3))


def test_l1potts_invalid_weights_negative():
    with pytest.raises(ValueError):
        L1Potts().fit(
            np.array([0.0, 0.0, 1.0, 1.0]),
            weights=np.array([1.0, -1.0, 1.0, 1.0]),
        )


def test_l1potts_int_dtype_signal():
    bkps = L1Potts().fit(np.array([0, 0, 5, 5, 5, -2, -2])).predict(pen=0.5)
    assert bkps[-1] == 7
    assert len(bkps) >= 2


# --- hardening: hostile and pathological inputs ---


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_l1potts_rejects_nonfinite_signal(bad):
    with pytest.raises(ValueError, match="finite"):
        L1Potts().fit(np.array([1.0, bad, 3.0]))


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_l1potts_rejects_nonfinite_weights(bad):
    with pytest.raises(ValueError, match="finite"):
        L1Potts().fit(np.array([1.0, 2.0, 3.0]), weights=np.array([1.0, bad, 1.0]))


def test_l1potts_rejects_complex_signal():
    with pytest.raises(ValueError, match="numeric"):
        L1Potts().fit(np.array([1 + 0j, 2 + 0j, 3 + 0j]))


def test_l1potts_rejects_object_signal():
    with pytest.raises(ValueError, match="numeric"):
        L1Potts().fit(np.array([1, 2, 3], dtype=object))


def test_l1potts_rejects_bool_signal():
    with pytest.raises(ValueError, match="numeric"):
        L1Potts().fit(np.array([True, False, True]))


def test_l1potts_rejects_complex_weights():
    with pytest.raises(ValueError, match="numeric"):
        L1Potts().fit(
            np.array([1.0, 2.0, 3.0]),
            weights=np.array([1 + 0j, 1 + 0j, 1 + 0j]),
        )


def test_l1potts_rejects_empty_signal():
    with pytest.raises(BadSegmentationParameters, match="non-empty"):
        L1Potts().fit(np.array([]))


def test_l1potts_rejects_empty_signal_2d():
    with pytest.raises(BadSegmentationParameters, match="non-empty"):
        L1Potts().fit(np.zeros((0, 1)))


def test_l1potts_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        L1Potts().predict(pen=0.5)


@pytest.mark.parametrize("bad_pen", [None, [0.5], object()])
def test_l1potts_rejects_non_numeric_pen(bad_pen):
    algo = L1Potts().fit(np.array([0.0, 1.0, 2.0]))
    with pytest.raises(ValueError):
        algo.predict(pen=bad_pen)


def test_l1potts_accepts_python_list_signal():
    """List/tuple inputs should silently work via np.asarray."""
    bkps = L1Potts().fit([0.0, 0.0, 5.0, 5.0]).predict(pen=0.5)
    assert bkps == [2, 4]


def test_l1potts_accepts_tuple_signal():
    bkps = L1Potts().fit((0.0, 0.0, 5.0, 5.0)).predict(pen=0.5)
    assert bkps == [2, 4]


def test_l1potts_accepts_python_list_weights():
    bkps = (
        L1Potts()
        .fit(
            np.array([1.0, 1.0, 100.0, 1.0, 1.0]),
            weights=[1.0, 1.0, 1e-3, 1.0, 1.0],
        )
        .predict(pen=10.0)
    )
    assert bkps == [5]


def test_l1potts_accepts_int_dtype_weights():
    bkps = (
        L1Potts()
        .fit(np.array([0.0, 0.0, 5.0, 5.0]), weights=np.array([1, 1, 1, 1]))
        .predict(pen=0.5)
    )
    assert bkps == [2, 4]


def test_l1potts_accepts_readonly_signal():
    sig = np.array([0.0, 0.0, 5.0, 5.0])
    sig.setflags(write=False)
    bkps = L1Potts().fit(sig).predict(pen=0.5)
    assert bkps == [2, 4]


def test_l1potts_accepts_noncontiguous_signal():
    # Stride trick: every other element of a length-8 array
    full = np.array([0.0, 99.0, 0.0, 99.0, 5.0, 99.0, 5.0, 99.0])
    sig = full[::2]
    assert not sig.flags["C_CONTIGUOUS"]
    bkps = L1Potts().fit(sig).predict(pen=0.5)
    assert bkps == [2, 4]


def test_l1potts_accepts_float32_signal():
    sig = np.array([0.0, 0.0, 5.0, 5.0], dtype=np.float32)
    bkps = L1Potts().fit(sig).predict(pen=0.5)
    assert bkps == [2, 4]


def test_l1potts_pen_inf_collapses_to_one_segment():
    """Pen=+inf should disallow all jumps and produce a single segment."""
    sig = np.array([0.0, 5.0, 0.0, 5.0, 0.0])
    bkps = L1Potts().fit(sig).predict(pen=float("inf"))
    assert bkps == [5]


def test_l1potts_pen_numpy_scalar():
    sig = np.array([0.0, 0.0, 5.0, 5.0])
    bkps_a = L1Potts().fit(sig).predict(pen=np.float64(0.5))
    bkps_b = L1Potts().fit(sig).predict(pen=np.array(0.5))
    assert bkps_a == [2, 4]
    assert bkps_b == [2, 4]


def test_l1potts_constant_K_equals_one():
    """K=1 (all unique values are identical) is a valid degenerate case."""
    bkps = L1Potts().fit(np.full(10, 7.0)).predict(pen=0.5)
    assert bkps == [10]


def test_l1potts_n_equals_two():
    sig = np.array([0.0, 5.0])
    bkps_low = L1Potts().fit(sig).predict(pen=0.1)
    bkps_high = L1Potts().fit(sig).predict(pen=1000.0)
    assert bkps_low == [1, 2]
    assert bkps_high == [2]


def test_l1potts_mutating_input_does_not_corrupt_fit():
    """After fit(), mutating the original signal array must not change the
    stored level set or reproducibility of predict()."""
    sig = np.array([0.0, 0.0, 5.0, 5.0])
    algo = L1Potts().fit(sig)
    bkps_before = algo.predict(pen=0.5)
    sig[:] = 0.0  # mutate the user's array
    bkps_after = algo.predict(pen=0.5)
    # The same segmentation must be returned even if the user mutates the
    # input. (We accept that ruptures convention is to share the array; this
    # test pins the actual contract: the algorithm doesn't recompute levels.)
    assert bkps_before == bkps_after


def test_l1potts_squeezes_only_trailing_singleton():
    """A (1, N) signal — leading singleton — should be accepted via squeeze."""
    bkps = L1Potts().fit(np.array([[0.0, 0.0, 5.0, 5.0]])).predict(pen=0.5)
    assert bkps == [2, 4]


def test_l1potts_runtime_warnings_silenced_on_clean_input(recwarn):
    """A clean numeric input must not emit RuntimeWarning."""
    L1Potts().fit(np.array([0.0, 0.0, 5.0, 5.0])).predict(pen=0.5)
    rt = [w for w in recwarn.list if issubclass(w.category, RuntimeWarning)]
    assert rt == [], [str(w.message) for w in rt]


def test_l1potts_compiles_without_syntax_warning():
    """Re-compile the module source under SyntaxWarning-as-error.

    Bytecode caching makes a runtime ``import`` insufficient — we must
    re-parse the source. Catches regression of unraw docstrings with
    backslash escapes (e.g. ``\\sum``) reintroducing SyntaxWarning at
    every consumer's first import.
    """
    import ruptures.detection.l1potts as mod

    with open(mod.__file__, "r") as fh:
        src = fh.read()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        compile(src, mod.__file__, "exec")
    syntax = [w for w in caught if issubclass(w.category, SyntaxWarning)]
    assert syntax == [], [str(w.message) for w in syntax]


def test_ruptures_imports_clean_in_subprocess():
    """``import ruptures`` must not emit any SyntaxWarning, in a fresh
    interpreter with -W error::SyntaxWarning. End-to-end safety net."""
    result = subprocess.run(
        [sys.executable, "-W", "error::SyntaxWarning", "-c", "import ruptures"],
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr.decode()
