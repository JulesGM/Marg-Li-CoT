import rich.console as _console

import general_utils as utils

# Ref https://rich.readthedocs.io/en/stable/console.html#console-api


class Console():
    def __init__(self, *args, **kwargs):
        self.__dict__["_console"] = _console.Console(*args, **kwargs)

    def __getattr__(self, name):
        if name not in self.__dict__:
            return getattr(self.__dict__["_console"], name)

    def __setattr__(self, name, value):
        if name in self.__dict__:
            self.__dict__[name] = value
        else:
            setattr(self._console, name, value)

    def __dir__(self):
        return list(self.__dict__.keys()) + list(dir(self._console))

    def print_zero_rank(self, *args, **kwargs):
        if utils.is_rank_zero():
            self.print(*args, **kwargs)

    def log_zero_rank(self, *args, **kwargs):
        if utils.is_rank_zero():
            self.log(*args, **kwargs)
