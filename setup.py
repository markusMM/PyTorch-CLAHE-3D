import os
import torch
from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, Extension

# Retrieve OpenCV include and library directories from environment variables
opencv_include = os.getenv('OPENCV_PATH', '/usr/local/include/opencv4')
opencv_lib = os.getenv('OPENCV_LIB', '/usr/local/lib')

# CUDA specific configuration
extra_compile_args = {'cxx': ['-O2']}
define_macros = []

# Check if CUDA is available
if torch.cuda.is_available():
    define_macros += [('WITH_CUDA', None)]
    extra_compile_args['nvcc'] = ['-O2']
    extensions = [
        CUDAExtension(
            name='compute_clahe',
            sources=['compute_clahe.cpp'],
            include_dirs=[opencv_include],
            library_dirs=[opencv_lib],
            libraries=['opencv_core', 'opencv_imgproc'],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
else:
    extensions = [
        Extension(
            name='compute_clahe',
            sources=['compute_clahe.cpp'],
            include_dirs=[opencv_include],
            library_dirs=[opencv_lib],
            libraries=['opencv_core', 'opencv_imgproc'],
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]

setup(
    name='clahe_extension',
    version='0.1',
    author='Your Name',
    packages=find_packages(),
    ext_modules=extensions,
    cmdclass={'build_ext': BuildExtension},
)
