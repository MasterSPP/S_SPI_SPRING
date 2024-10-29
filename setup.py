import os
import re
import sys
import platform
import subprocess

import setuptools
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion


if os.environ.get('CI_COMMIT_TAG'):
    version = os.environ['CI_COMMIT_TAG']
else:
    version = "0.1"

name = "spring_core"


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        # required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable ]

        #cfg = 'Debug' if self.debug else 'Release'
        cfg = 'Release'

        build_args = ['--config', cfg]

        cmake_args += ['-DCMAKE_POSITION_INDEPENDENT_CODE=True']
        if platform.system() == "Windows":
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            cmake_args += ['-DCMAKE_MSVC_RUNTIME_LIBRARY=MultiThreaded']
            cmake_args += ['-DCMAKE_CUDA_RUNTIME_LIBRARY=Static']
            if sys.maxsize > 2**32:
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m', '/verbosity:normal']
        else:
            cmake_args += ['-DCMAKE_CUDA_RUNTIME_LIBRARY=Static']
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            #build_args += ['--', '-j4', '-Wl,--no-undefined']
            build_args += ['--', '-j4']

        if os.environ.get('WITH_CUDA')=="Off":
            cmake_args += ['-DWITH_CUDA=Off']
        else:
            cmake_args += ['-DWITH_CUDA=On']
        
        if os.environ.get('WITH_ARCH')=="Off":
            cmake_args += ['-DWITH_ARCH=Off']
        else:
            cmake_args += ['-DWITH_ARCH=On']

        if os.environ.get('WITH_NVML')=="On":
            cmake_args += ['-DWITH_NVML=On']
        else:
            cmake_args += ['-DWITH_NVML=Off']
            
            
        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(env.get('CXXFLAGS', ''),
                                                              self.distribution.get_version())
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(['cmake', ext.sourcedir] + cmake_args, self.build_temp, env)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name="spring",
    version=version,
    author='Alessandro Colombo',
    author_email='alcolombo@phys.ethz.ch',
    description='A software for data analysis of single-shot single-particle Free Electron Laser CDI analysis',
    long_description='',
    scripts=[],
    ext_modules=[CMakeExtension(name)],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    packages = setuptools.find_packages(),
    install_requires=[
   'numpy',
   'h5py',
   'PyYAML',
   'matplotlib',
   'prettytable',
   'IPython'
    ]
)
