from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(
    name='clahe3d',
    ext_modules=[
        cpp_extension.CppExtension(
            name='torch_clahe',
            sources=['torch_clahe.cpp'],
            extra_compile_args={'cxx': ['-std=c++17']}
        )
    ],
    cmdclass={'build_ext': cpp_extension.BuildExtension},
    version='0.99.19',
    requires=["torch"]
)

# Extension(
#     name='clahe3d',
#     sources=['torch_clahe.cpp'],
#     include_dirs=cpp_extension.include_paths(),
#     language='c++'
# )
