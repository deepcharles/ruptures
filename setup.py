from setuptools import Extension, setup


# try:
# from Cython.Distutils.build_ext import new_build_ext as build_ext
# except ImportError:
# from setuptools.command.build_ext import build_ext

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


# cmdclass = dict()
# cmdclass["build_ext"] = build_ext

# setup(
# cmdclass=cmdclass,
# ext_modules=ext_modules,
# )

# https://luminousmen.com/post/resolve-cython-and-numpy-dependencies
try:
    from Cython.Build import cythonize
except ImportError:
    # create closure for deferred import
    def cythonize(*args, **kwargs):
        from Cython.Build import cythonize

        return cythonize(*args, **kwargs)


setup(
    setup_requires=[
        # Setuptools 18.0 properly handles Cython extensions.
        "setuptools>=18.0",
        "cython",
    ],
    ext_modules=cythonize(ext_modules, language_level="3"),
)
