from .context_timer import ContextTimer

import statistics as stats

class ContextTimerManager:
  def __init__(self,comm):
    self.comm = comm
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

    str_format = "  {name:<{width}} || {mean:^16.4e} | {stdev:^16.4e} |\n" 

    result = ""
    result += "  {name:^{width}} || {mean:^16} | {stdev:^16} |\n".format(name="timer",mean="mean",stdev="stdev",width=max_width)
    result += "======================================================\n"
    for name,timer in self.timers.items():
      mean  = stats.mean(timer.getTimes())
      stdev = stats.stdev(timer.getTimes())
      result += str_format.format(name=name,mean=mean,stdev=stdev,width=max_width)

    return result
  # end getResultString

# end ContextTimerManager
