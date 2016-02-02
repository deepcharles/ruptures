from setuptools import setup, find_packages

setup(
    name='ruptures',
    version='0.0.1',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'matplotlib'],
    url='',
    license='je ne sais pas',
    author='charles',
    author_email='charles.truong@cmla.ens-cachan.fr',
    description='Several segmentation algorithms',
    download_url='',
    keywords=["Miscellaneous", "Toy data generator"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Topic :: Computer Science",
    ],
    long_description="""\
Miscellaneous functions.
-------------------------------------
This package contains functions to generate synthetic data.

This version requires Python 3 or later.
"""
)
