import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

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
    ext_modules=cythonize(
        ext_modules,
        language_level="3",
    ),
)
