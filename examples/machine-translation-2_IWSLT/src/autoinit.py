import inspect

def AutoInitializer(Base):
  class Wrapper(Base):
    def __init__(self, *args, **provided_kwargs):
      parameters = inspect.signature(Base).parameters
      arg_nms = parameters.keys()
      kwargs = {
        k: v.default for (k, v) in list(parameters.items())[len(args):]
      }
      kwargs.update(provided_kwargs)
      missing_arguments = list(filter(
        lambda k: kwargs[k] == inspect._empty, kwargs.keys()
      ))
      if missing_arguments:
        missing_arguments = [f"'{arg}'" for arg in missing_arguments]
        error_text = ''.join([
          f"__init__() missing {len(missing_arguments)} required positional ",
          "argument" if len(missing_arguments) == 1 else "arguments", 
          ": ",
          ' and '.join(missing_arguments) if len(missing_arguments) <= 2 else \
          ', '.join(missing_arguments[:-1]) + f', and {missing_arguments[-1]}'
          ])
        raise TypeError(error_text)

      for arg_nm, arg in zip(arg_nms, args): self.__dict__[arg_nm] = arg
      for (k, v) in kwargs.items()         : self.__dict__[  k   ] = v

      super().__init__(*args, **kwargs)

    def __repr__(self):
      if '__repr__' in Base.__dict__: 
        #return Base.__repr__()
        self.__dict__['__repr__'] = Base.__dict__['__repr__']
        return self.__repr__(self)# + ' | (AutoInitializer wrapper)'
      else:
        return ' '.join([
          f"<autoinit.AutoInitializer.<locals>.Wrapper> derived from",
          f"{Base.__module__}.{Base.__name__}",
        ])

  return Wrapper



