import os
from setuptools import Extension, setup
from setuptools_scm import get_version

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

    version_file = os.path.join("src", "ruptures", "version.py")
    if not os.path.exists(version_file):
        version = get_version(root='.', relative_to=__file__)
        if not version:
            raise RuntimeError("Version could not be determined. Ensure you have a valid git tag.")
        with open(version_file, "w") as f:
             f.write(f'__version__ = version = "{version}"\n')

    setup(
        ext_modules=cythonize(ext_modules, language_level="3"),
    )
