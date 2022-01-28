from setuptools import Extension, setup

ext_modules = [
    Extension(
        "ruptures.detection._detection.ekcpd",
        sources=[
            "src/ruptures/detection/_detection/ekcpd.pyx",
            "src/ruptures/detection/_detection/ekcpd_computation.c",
            "src/ruptures/detection/_detection/ekcpd_pelt_computation.c",
            "src/ruptures/detection/_detection/kernels.c",
        ],
    ),
    Extension(
        "ruptures.utils._utils.convert_path_matrix",
        sources=[
            "src/ruptures/utils/_utils/convert_path_matrix.pyx",
            "src/ruptures/utils/_utils/convert_path_matrix_c.c",
        ],
    ),
]


if __name__ == "__main__":
    from Cython.Build import cythonize

    setup(
        ext_modules=cythonize(ext_modules, language_level="3"),
    )
