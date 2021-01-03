from setuptools import Extension, find_packages, setup

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


if __name__ == "__main__":
    from Cython.Build import cythonize

    setup(
        packages=find_packages(exclude=["docs", "tests*", "images"]),
        ext_modules=cythonize(ext_modules, language_level="3"),
    )
