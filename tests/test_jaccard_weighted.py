try:
    import unittest2 as unittest
except ImportError:
    import unittest
import distpy
import numpy as np
# Cheat Sheet (method/test) <http://docs.python.org/library/unittest.html>
#
# assertEqual(a, b)       a == b   
# assertNotEqual(a, b)    a != b    
# assertTrue(x)     bool(x) is True  
# assertFalse(x)    bool(x) is False  
# assertRaises(exc, fun, *args, **kwds) fun(*args, **kwds) raises exc
# assertAlmostEqual(a, b)  round(a-b, 7) == 0         
# assertNotAlmostEqual(a, b)          round(a-b, 7) != 0
# 
# Python 2.7+ (or using unittest2)
#
# assertIs(a, b)  a is b
# assertIsNot(a, b) a is not b
# assertIsNone(x)   x is None
# assertIsNotNone(x)  x is not None
# assertIn(a, b)      a in b
# assertNotIn(a, b)   a not in b
# assertIsInstance(a, b)    isinstance(a, b)
# assertNotIsInstance(a, b) not isinstance(a, b)
# assertRaisesRegexp(exc, re, fun, *args, **kwds) fun(*args, **kwds) raises exc and the message matches re
# assertGreater(a, b)       a > b
# assertGreaterEqual(a, b)  a >= b
# assertLess(a, b)      a < b
# assertLessEqual(a, b) a <= b
# assertRegexpMatches(s, re) regex.search(s)
# assertNotRegexpMatches(s, re)  not regex.search(s)
# assertItemsEqual(a, b)    sorted(a) == sorted(b) and works with unhashable objs
# assertDictContainsSubset(a, b)      all the key/value pairs in a exist in b


def jaccard_weighted_slow(a, b, w):
    out = np.double(0.)
    for x in range(w.size / 8):
        cur_w = w[8*x:8*(x+1)]
        out += sum(cur_w[np.unpackbits(a[x] & b[x]).astype(np.bool)])
    return out


class Test(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_name(self):
        for n in [16, 1, 5, 8, 16, 20, 20 * 8]:
            w = np.random.random(n)
            j = distpy.JaccardWeighted(w)
            for x in range(16):
                a = np.zeros(n, dtype=np.uint8)
                b = np.zeros(n, dtype=np.uint8)
                a[x] = 1
                b[x] = 1
                a = np.packbits(a)
                b = np.packbits(b)
                out0 = j.dist(a, b)
                out1 = jaccard_weighted_slow(a, b, w)
                self.assertEqual(out0, out1)
                self.assertEqual(out0, w[x])
                print((out0, out1, w[x]))
            for x in range(1000):
                a = np.fromstring(np.random.bytes(n * 2), dtype=np.uint8)
                b = np.fromstring(np.random.bytes(n * 2), dtype=np.uint8)
                out0 = j.dist(a, b)
                out1 = jaccard_weighted_slow(a, b, w)
                self.assertAlmostEqual(out0, out1)
                print((out0, out1))

if __name__ == '__main__':
    import cProfile
    cProfile.run('unittest.main()', 'outprof')
    import pstats
    p = pstats.Stats('outprof')
    p.sort_stats('cumulative').print_stats(20)
