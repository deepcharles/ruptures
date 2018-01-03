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

This package contains:
    - algorithms to detect change points in signals,
    - functions to evaluate methods' performances,
    - functions to display the estimated change points,
    - function to generate toy signals.

This version requires Python 3 or later.
"""
)
