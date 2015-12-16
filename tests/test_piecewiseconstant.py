import numpy as np
import ruptures.dynamic_programming.piecewise_constant as pc


def test_2ruptures():
    n_samples = 200
    time = np.linspace(0, 12, n_samples)
    # 2 ruptures
    sig = np.sign(np.sin(0.7 * time))
    # vraies ruptures
    vraies_ruptures = [k + 1 for k,
                       x in enumerate(np.diff(sig)[1:], start=1) if x != 0]
    c = pc.Constant(sig)
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
        c = pc.Constant(sig)
        res = c.fit(len(vraies_ruptures) + 1, 1, 1)

        ruptures = [s for (s, e) in res.keys() if s != 0]
        ruptures.sort()
        print((ruptures, vraies_ruptures))
        assert vraies_ruptures == ruptures
