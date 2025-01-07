from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension("generation_sequence",
              sources=["generation_sequence.pyx"],
              libraries=["stdlib"]  # Unix-like specific
              )
]

setup(name="generation sequence",
      ext_modules=cythonize(ext_modules))