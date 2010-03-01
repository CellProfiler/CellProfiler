

class Array:
    def __init__(self, m, n):
#        self.type = type
        self.data = [[None] * n] * m
    def __getitem__(self, x):
        i, j = x
        return self.data[i][j]
    def __setitem__(self, x, value):
        i, j = x
        self.data[i][j] = value

class Struct:
    pass

if __name__ == "__main__":
    c = CellArray(4,5)
    c[3,4] = 42
    print c[3,4]
