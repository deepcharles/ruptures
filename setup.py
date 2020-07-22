from setuptools import setup, find_packages

setup(
    name='ruptures',
    version='1.0.4',
    packages=find_packages(exclude=['docs', 'tests*', 'images']),
    install_requires=['numpy', 'scipy'],
    extras_require={
        'display': ['matplotlib']
    },
    python_requires='>=3',
    # url='ctruong.perso.math.cnrs.fr/ruptures',
    license='BSD License',
    author='Charles Truong, Laurent Oudre, Nicolas Vayatis',
    author_email='truong@cmla.ens-cachan.fr',
    description='Change point detection for signals, in Python',
    download_url='https://reine.cmla.ens-cachan.fr/c.truong/ruptures/repository/latest/archive.zip',
    keywords=["change point detection", "signal segmentation",
              "computer science", "machine learning", "kernel methods", "time series"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License"
    ],
    long_description="""\
Offline change point detection for Python.
------------------------------------------

**ruptures** is a Python library for offline change point detection. This package provides methods for the analysis and segmentation of non-stationary signals. Implemented algorithms include exact and approximate detection for various parametric and non-parametric models. **ruptures** focuses on ease of use by providing a well-documented and consistent interface. In addition, thanks to its modular structure, different algorithms and models can be connected and extended within this package.


An extensive documentation is available `ctruong.perso.math.cnrs.fr/ruptures <http://ctruong.perso.math.cnrs.fr/ruptures>`_.

This version requires Python 3 or later.
"""
)
