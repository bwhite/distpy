import distutils.sysconfig
env = Environment()
env.Append(CFLAGS =  '-O3 -Wall -mmmx -msse -msse2 -msse3 -mssse3 -ftree-vectorizer-verbose=2')
env.SharedLibrary('distpy/lib/knearest_neighbor', ['distpy/knearest_neighbor.c'])