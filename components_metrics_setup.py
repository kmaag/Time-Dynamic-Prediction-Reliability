from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy as np

ext_core = Extension(
        "components_metrics",
        sources=["components_metrics.pyx"],
        include_dirs=[np.get_include()],
        extra_compile_args=["-O3"])
 
setup( ext_modules = cythonize([ext_core]) ) 
