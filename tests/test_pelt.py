import numpy as np
import ruptures.pelt.Pelt as p
import ruptures.pelt.costs as costs
import sklearn.datasets.samples_generator as s_generator


def test_ruptures1D():
    n_ruptures = 5
    n_samples, n_features = 500, 1
    stds = np.linspace(1, 20, n_ruptures)
    sig, y = s_generator.make_blobs(n_samples=n_samples, centers=n_ruptures,
                                    n_features=n_features,
                                    cluster_std=stds,
                                    shuffle=False)

    c = costs.gaussmean(sig)
    for pen in np.logspace(0.1, 100, 20):
        pe = p.Pelt(c, penalty=pen, n=sig.shape[0], K=0)
        pe.fit()
