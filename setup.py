from setuptools import setup, find_packages

setup(
    name='ruptures',
    version='1.0',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib'],
    url='',
    license='BSD License',
    author='Charles Truong',
    author_email='truong@cmla.ens-cachan.fr',
    description='Change point detection for Python',
    download_url='',
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
-------------------------------------

__ruptures__ is a Python library for offline change point detection. This package provides methods for the analysis and segmentation of non-stationary signals. Implemented algorithms include exact and approximate detection for various parametric and non-parametric models. __ruptures__ focuses on ease of use by providing a well-documented and consistent interface. In addition, thanks to its modular structure, different algorithms and models can be connected and extended within this package.

This version requires Python 3 or later.
"""
)
