# Welcome to ruptures

[![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/deepcharles/ruptures/graphs/commit-activity)
[![build](https://github.com/deepcharles/ruptures/actions/workflows/run-test.yml/badge.svg)](https://github.com/deepcharles/ruptures/actions/workflows/run-test.yml)
![python](https://img.shields.io/badge/python-3.6%20%7C%203.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue)
[![PyPI version](https://badge.fury.io/py/ruptures.svg)](https://badge.fury.io/py/ruptures)
[![Conda Version](https://img.shields.io/conda/vn/conda-forge/ruptures.svg)](https://anaconda.org/conda-forge/ruptures)
[![docs](https://github.com/deepcharles/ruptures/actions/workflows/check-docs.yml/badge.svg)](https://github.com/deepcharles/ruptures/actions/workflows/check-docs.yml)
![PyPI - License](https://img.shields.io/pypi/l/ruptures)
[![Downloads](https://pepy.tech/badge/ruptures)](https://pepy.tech/project/ruptures)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/deepcharles/ruptures/master)
[![Codecov](https://codecov.io/gh/deepcharles/ruptures/branch/master/graphs/badge.svg)](https://app.codecov.io/gh/deepcharles/ruptures/branch/master)

`ruptures` is a Python library for off-line change point detection.
This package provides methods for the analysis and segmentation of non-stationary signals.  Implemented algorithms include exact and approximate detection for various parametric and non-parametric models.
`ruptures` focuses on ease of use by providing a well-documented and consistent interface.
In addition, thanks to its modular structure, different algorithms and models can be connected and extended within this package.

**How to cite.** If you use `ruptures` in a scientific publication, we would appreciate citations to the following paper:

- C. Truong, L. Oudre, N. Vayatis. Selective review of offline change point detection methods. _Signal Processing_, 167:107299, 2020. [[journal]](https://doi.org/10.1016/j.sigpro.2019.107299) [[pdf]](http://www.laurentoudre.fr/publis/TOG-SP-19.pdf)


## Basic usage

(Please refer to the [documentation](https://centre-borelli.github.io/ruptures-docs/ "Link to documentation") for more advanced use.)

The following snippet creates a noisy piecewise constant signal, performs a penalized kernel change point detection and displays the results (alternating colors mark true regimes and dashed lines mark estimated change points).

```python
import matplotlib.pyplot as plt
import ruptures as rpt

# generate signal
n_samples, dim, sigma = 1000, 3, 4
n_bkps = 4  # number of breakpoints
signal, bkps = rpt.pw_constant(n_samples, dim, n_bkps, noise_std=sigma)

# detection
algo = rpt.Pelt(model="rbf").fit(signal)
result = algo.predict(pen=10)

# display
rpt.display(signal, bkps, result)
plt.show()
```

![](./images/example_readme.png)

## General information

#### Contact

Concerning this package, its use and bugs, use the [issue page](https://github.com/deepcharles/ruptures/issues) of the [ruptures repository](https://github.com/deepcharles/ruptures). For other inquiries, you can contact me [here](https://charles.doffy.net/contact/).


#### Important links

- [Documentation](https://centre-borelli.github.io/ruptures-docs)
- [Pypi package index](https://pypi.python.org/pypi/ruptures)

#### Dependencies and install

Installation instructions can be found [here](https://centre-borelli.github.io/ruptures-docs/install/).

#### Changelog

See the [changelog](https://github.com/deepcharles/ruptures/blob/master/CHANGELOG.md) for a history of notable changes to `ruptures`.

## Thanks to all our contributors

<a href="https://github.com/deepcharles/ruptures/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=deepcharles/ruptures" />
</a>

## License

This project is under BSD license.

```
BSD 2-Clause License

Copyright (c) 2017-2022, ENS Paris-Saclay, CNRS
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
