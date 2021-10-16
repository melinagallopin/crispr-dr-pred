from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


extensions = [
    Extension("find_new_clusters", sources=["find_new_clusters.pyx"],
              include_dirs=[np.get_include()], extra_compile_args=["-O3"], language="c++")
]

setup(
    name="find_new_clusters",
    ext_modules=cythonize(extensions, language_level="3"),
)

extensions = [
    Extension("probabilistic_classifier_cython", sources=["probabilistic_classifier_cython.pyx"],
              include_dirs=[np.get_include()], extra_compile_args=["-O3"], language="c++")
]

setup(
    name="probabilistic_classifier_cython",
    ext_modules=cythonize(extensions, language_level="3"),
)


# import pyximport; pyximport.install(setup_args={"include_dirs": np.get_include()},
#                                     language_level="3", reload_support=True)
