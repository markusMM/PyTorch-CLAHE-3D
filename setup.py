from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='clahe3d',
    ext_modules=[
        cpp_extension.CppExtension('torch_clahe', ['torch_clahe.cpp'])
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    version='0.98.0',
    requires=["torch"]
)

Extension(
    name='clahe3d',
    sources=['torch_clahe.cpp'],
    include_dirs=cpp_extension.include_paths(),
    language='c++'
)
