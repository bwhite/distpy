from distutils.core import setup
import re
from distutils.extension import Extension
import numpy as np


def get_cython_version():
    """
    Returns:
        Version as a pair of ints (major, minor)

    Raises:
        ImportError: Can't load cython or find version
    """
    import Cython.Compiler.Main
    match = re.search('^([0-9]+)\.([0-9]+)',
                      Cython.Compiler.Main.Version.version)
    try:
        return map(int, match.groups())
    except AttributeError:
        raise ImportError

# Only use Cython if it is available, else just use the pre-generated files
try:
    cython_version = get_cython_version()
    # Requires Cython version 0.13 and up
    if cython_version[0] == 0 and cython_version[1] < 13:
        raise ImportError
    from Cython.Distutils import build_ext
    source_ext = '.pyx'
    cmdclass = {'build_ext': build_ext}
except ImportError:
    source_ext = '.c'
    cmdclass = {}

#  '-ftree-vectorizer-verbose=2'
ext_modules = [Extension("_distpy_l2",
                         ["distpy/l2" + source_ext,
                          'distpy/knearest_neighbor.c'],
                         extra_compile_args=['-I', np.get_include(),
                                             '-O3', '-Wall', '-mmmx', '-msse',
                                             '-msse2'])]

setup(name='distpy',
      cmdclass=cmdclass,
      version='.01',
      packages=['distpy'],
      ext_modules=ext_modules)
