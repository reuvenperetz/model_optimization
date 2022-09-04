import unittest
import time


class TimerTestCase(unittest.TestCase):

    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('Test runtime (sec): %s: %.3f' % (self.id(), t))
