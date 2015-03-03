# -*- coding: utf-8 -*-
__author__ = 'Adam Kaczmarek, Filip Mróz'

import pickle


class PickleSnakeListFormat(object):

    def __init__(self, segmentation, snakes):
        self.segmentation = segmentation
        self.snakes = snakes

    def read(self, filename):
        src = open(filename, "r")
        p = pickle.Unpickler(src)
        self.snakes = p.load()
        src.close()

    def write(self, filename):
        print filename
        dst = open(filename, "w")
        p = pickle.Pickler(dst)
        p.dump(self.snakes)
        dst.close()