from distutils.core import setup

setup(name='distpy',
      version='.01',
      packages=['distpy'],
      package_data = {'distpy': ['lib/*.so']}
      )