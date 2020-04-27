from .context_timer import ContextTimer

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
# end ContextTimerManager
