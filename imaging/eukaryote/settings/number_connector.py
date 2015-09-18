class NumberConnector(object):
    '''This object connects a function to a number slot

    You can use this if you have a value that changes contextually
    depending on other settings. You pass in a function that, when evaluated,
    gives the current value for the number. You can then pass in a number
    connector instead of an explicit value for things like minima and maxima
    for numeric settings.
    '''

    def __init__(self, fn):
        self.__fn = fn

    def __int__(self):
        return int(self.__fn())

    def __long__(self):
        return long(self.__fn())

    def __float__(self):
        return float(self.__fn())

    def __cmp__(self, other):
        return cmp(self.__fn(), other)

    def __hash__(self):
        return self.__fn().__hash__()

    def __str__(self):
        return str(self.__fn())
