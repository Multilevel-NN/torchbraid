from timeit import default_timer as timer

class ContextTimer:
  def __init__(self,name):
    self.name   = name
    self.times  = []
    self.timing = False 

  def __enter__(self):
    self.timing = True
    self.start_time = timer()
    return self

  def __exit__(self,except_type,except_value,except_traceback):
    self.end_time = timer()
    self.timing = False

    self.times += [ self.end_time-self.start_time ] 
    return True

  def getName(self):
    return self.name

  def isTiming(self):
    return self.timing

  def getTimes(self):
    return self.times
