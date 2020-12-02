from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
import numpy as np
from Cython.Build import cythonize


ext_modules = [
    Extension(
        "ruptures.detection._detection.ekcpd",
        sources=[
            "ruptures/detection/_detection/ekcpd.pyx",
            "ruptures/detection/_detection/ekcpd_computation.c",
            "ruptures/detection/_detection/ekcpd_pelt_computation.c",
            "ruptures/detection/_detection/kernels.c",
        ],
    ),
    Extension(
        "ruptures.utils._utils.convert_path_matrix",
        sources=[
            "ruptures/utils/_utils/convert_path_matrix.pyx",
            "ruptures/utils/_utils/convert_path_matrix_c.c",
        ],
    ),
]

setup(
    name="ruptures",
    version="1.1.2rc1",
    packages=find_packages(exclude=["docs", "tests*", "images"]),
    install_requires=["numpy", "scipy"],
    extras_require={"display": ["matplotlib"]},
    setup_requires=["cython", "numpy"],
    package_data={
        "ruptures.detection._detection": [
            "ekcpd.pxd",
            "ekcpd.pyx",
            "ekcpd_computation.h",
            "ekcpd_pelt_computation.h",
            "kernels.h",
        ],
        "ruptures.utils._utils": [
            "convert_path_matrix.pyx",
            "convert_path_matrix.pxd",
            "convert_path_matrix_c.h",
        ],
    },
    cmdclass={"build_ext": build_ext},
    python_requires=">=3",
    url="https://centre-borelli.github.io/ruptures-docs/",
    license="BSD License",
    author="Charles Truong, Laurent Oudre, Nicolas Vayatis",
    author_email="charles@doffy.net",
    maintainer="Charles Truong, Olivier Boulant",
    description="Change point detection for signals, in Python",
    download_url="https://github.com/deepcharles/ruptures/archive/master.zip",
    keywords=[
        "change point detection",
        "signal segmentation",
        "computer science",
        "machine learning",
        "kernel methods",
        "time series",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
    ],
    long_description="""\
Offline change point detection for Python.
------------------------------------------

**ruptures** is a Python library for offline change point detection.
This package provides methods for the analysis and segmentation of
non-stationary signals.
Implemented algorithms include exact and approximate detection for various
parametric and non-parametric models.
**ruptures** focuses on ease of use by providing a well-documented and
consistent interface. In addition, thanks to its modular structure, different
algorithms and models can be connected and extended within this package.


An extensive documentation is available
`github.com/deepcharles/ruptures <https://github.com/deepcharles/ruptures>`_.

This version requires Python 3 or later.
""",
    ext_modules=cythonize(
        ext_modules,
        language_level="3",
    ),
)
