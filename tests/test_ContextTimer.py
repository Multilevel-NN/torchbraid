import unittest
import faulthandler
faulthandler.enable()

import time
import torchbraid.utils as utils

class TestContextTimer(unittest.TestCase):

  def test_ContextTiming(self):

     comm = None
     mgr = utils.ContextTimerManager(comm)
     clock = mgr.timer("hello")

     self.assertTrue(not clock.isTiming())
     self.assertTrue(clock.getName()=="hello")
     self.assertTrue(len(clock.getTimes())==0)

     for i in range(5):
       clock_timing_in_context = None
       with clock:
         clock_timing_in_context = clock.isTiming() 
         time.sleep(0.10) 
       self.assertTrue(clock_timing_in_context)

     self.assertTrue(not clock.isTiming())
     self.assertTrue(clock.getName()=="hello")
     self.assertTrue(len(clock.getTimes())==5)

     self.assertTrue(len(mgr.getTimers())==1)

     for i in range(5):
       clock_timing_in_context = None
       with mgr.timer("cat") as clock_2:
         clock_timing_in_context = clock_2.isTiming() 
         time.sleep(0.10) 
         clock_save = clock_2
       self.assertTrue(clock_timing_in_context)

     self.assertTrue(not clock_save.isTiming())
     self.assertTrue(clock_save.getName()=="cat")
     self.assertTrue(len(clock_save.getTimes())==5)

     self.assertTrue(len(mgr.getTimers())==2)
  # end test_ContextTiming(self):
# end TestTimerContext

if __name__ == '__main__':
  unittest.main()
