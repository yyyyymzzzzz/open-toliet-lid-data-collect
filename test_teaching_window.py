import unittest
from manual_motion import TeachingWindow,MotionError,pre_roll_seconds

class WindowTests(unittest.TestCase):
    def test_long_wait_keeps_only_one_second_before_motion(self):
        w=TeachingWindow(1)
        for i in range(3600):w.add(i/30,[.002*(i%2)]*6)
        self.assertFalse(w.moved);self.assertLessEqual(len(w.waiting),32)
        self.assertEqual(w.samples,[])
        w.add(120,[.31]*6)
        self.assertTrue(w.moved)
        self.assertAlmostEqual(w.origin,119)
        self.assertEqual(w.samples[0]['t'],0)
        self.assertAlmostEqual(w.samples[-1]['t'],1)
        w.add(120+1/30,[.32]*6)
        self.assertAlmostEqual(w.samples[-1]['t'],1+1/30)

    def test_zero_and_custom_pre_roll(self):
        for pre in (0,.5,2,10):
            w=TeachingWindow(pre)
            for i in range(601):w.add(i/30,[0]*6)
            w.add(20+1/30,[1]*6)
            self.assertLessEqual(w.samples[-1]['t'],pre+1/30+1e-9)
            self.assertGreaterEqual(w.samples[-1]['t'],pre-1e-9)
            self.assertEqual(w.samples[0]['t'],0)

    def test_wait_does_not_consume_duration_limit(self):
        w=TeachingWindow(1)
        for i in range(10000):w.add(i/30,[0]*6)
        w.add(10000/30,[1]*6)
        w.add(w.origin+140,[1]*6)
        with self.assertRaises(MotionError):w.add(w.origin+140.01,[1]*6)

    def test_trailing_stillness_keeps_one_second_after_motion(self):
        w=TeachingWindow(1)
        w.add(0,[0]*6);w.add(0.1,[1]*6)
        for i in range(1,301):w.add(0.1+i/30,[1]*6)
        w.trim_tail(1)
        self.assertAlmostEqual(w.samples[-1]['t'],1.0,places=2)
        self.assertGreater(w.trimmed_trailing_s,8)

    def test_invalid_setting_rejected(self):
        for value in (-1,10.1,float('nan'),float('inf'),True,'1',None):
            with self.assertRaises(MotionError):pre_roll_seconds(value)

if __name__=='__main__':unittest.main()
