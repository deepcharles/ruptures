import numpy as np
import ruptures.dynamic_programming.piecewise_kernel as pk


def test_2ruptures():
    n_samples = 200
    time = np.linspace(0, 12, n_samples)
    # 2 ruptures
    sig = np.sign(np.sin(0.7 * time))
    # vraies ruptures
    vraies_ruptures = [k + 1 for k,
                       x in enumerate(np.diff(sig)[1:], start=1) if x != 0]
    c = pk.Kernel(sig, "linear")
    res = c.fit(len(vraies_ruptures) + 1, 1, 1)

    ruptures = [s for (s, e) in res.keys() if s != 0]
    ruptures.sort()
    print((ruptures, vraies_ruptures))
    assert vraies_ruptures == ruptures


def test_ruptures():
    n_samples = 500
    time = np.linspace(0, 12, n_samples)
    for mult in np.linspace(0.5, 2, 5):
        sig = np.sign(np.sin(mult * time))
        # vraies ruptures
        vraies_ruptures = [k + 1 for k,
                           x in enumerate(np.diff(sig)[1:], start=1) if x != 0]
        c = pk.Kernel(sig, "linear")
        res = c.fit(len(vraies_ruptures) + 1, 1, 1)

        ruptures = [s for (s, e) in res.keys() if s != 0]
        ruptures.sort()
        print((ruptures, vraies_ruptures))
        assert vraies_ruptures == ruptures


def test_kernels():
    n_samples = 500
    time = np.linspace(0, 12, n_samples)
    mult = 1.1
    sig = np.sign(np.sin(mult * time)).reshape(-1, 1)
    a = np.arange(10).reshape(-1, 1)
    dummy = np.column_stack([a] * n_samples).T
    sig = np.column_stack((sig, dummy))
    assert sig.shape[0] == n_samples
    vraies_ruptures = [k + 1 for (k, (v, w)) in
                       enumerate(zip(sig[:-1], sig[1:]))
                       if not np.allclose(v, w)]
    for ker in pk.valid_kernels:
        c = pk.Kernel(sig, kernel=ker)
        res = c.fit(len(vraies_ruptures) + 1, 1, 1)

        ruptures = [s for (s, e) in res.keys() if s != 0]
        ruptures.sort()
        print((ruptures, vraies_ruptures))
        assert vraies_ruptures == ruptures
