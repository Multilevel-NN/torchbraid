from .context_timer import ContextTimer

import statistics as stats

class ContextTimerManager:
  def __init__(self):
    self.timers = dict()

  def timer(self,name): 
    if name in self.timers:
      timer_obj = self.timers[name]
    else:
      timer_obj = ContextTimer(name) 
      self.timers[name] = timer_obj

    return timer_obj
  # end timer

  def getTimers(self):
    return list(self.timers.values())
 
  def getResultString(self):
    max_width = len("name")
    for name,timer in self.timers.items():
      max_width = max(max_width,len(name))

    str_format = "  {name:<{width}} || {count:^16d} | {total:^16.4e} | {mean:^16.4e} | {stdev:^16.4e} |\n" 

    result = ""
    result +=    "  {name:^{width}} || {count:^16} | {total:^16} | {mean:^16} | {stdev:^16} |\n".format(name="timer",
                                                                                          count="count",
                                                                                          total="total",
                                                                                          mean="mean",
                                                                                          stdev="stdev",
                                                                                          width=max_width)
    result += "======================================================\n"

    keys = list(self.timers.keys())
    keys.sort()
    for name in keys:
      timer = self.timers[name]
      times = timer.getTimes()
      mean  = stats.mean(times)
      total  = sum(times)

      if len(times)>1:
        stdev = stats.stdev(times)
      else:
        stdev = 0.0

      result += str_format.format(name=name,count=len(times),total=total,mean=mean,stdev=stdev,width=max_width)

    return result
  # end getResultString

# end ContextTimerManager
