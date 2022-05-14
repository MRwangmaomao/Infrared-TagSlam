from distutils.core import Extension, setup
from Cython.Build import cythonize

# define an extension that will be cythonized and compiled
ext1 = Extension(name="CFNS", sources=["CFNS.pyx"])
setup(ext_modules=cythonize(ext1))

# command to build extension is 'python3 CONV.py build_ext --inplace'