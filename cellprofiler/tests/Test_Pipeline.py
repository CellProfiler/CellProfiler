"""test_Pipeline.py - test the CellProfiler.Pipeline module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2011 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
"""
__version__ = "$Revision: 1$"

import os

import base64
import unittest
import numpy as np
import numpy.lib.index_tricks
import cStringIO
import zlib

import cellprofiler.pipeline as cpp
import cellprofiler.objects as cpo
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.settings as cps
import cellprofiler.measurements as cpmeas
import cellprofiler.workspace as cpw
import cellprofiler.modules
from cellprofiler.modules.injectimage import InjectImage

def module_directory():
    d = cpp.__file__
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler/cellProfiler
    d = os.path.split(d)[0] # ./CellProfiler/pyCellProfiler
    d = os.path.split(d)[0] # ./CellProfiler
    if not d:
        d = '..'
    return os.path.join(d,'Modules')

def image_with_one_cell(size=(100,100)):
    img = np.zeros(size)
    mgrid = np.lib.index_tricks.nd_grid()
    g=mgrid[0:100,0:100]-50                              # the manhattan distance from 50,50
    dist = g[0,:,:]*g[0,:,:]+g[1,:,:]*g[1,:,:]           # the 2-d distance (forgot the fancy name)
    img[ dist < 25] = (25.0-dist.astype(float)[dist<25])/25 # A circle in the middle of it
    return img

def exploding_pipeline(test):
    """Return a pipeline that fails if the run exception callback is called during a run
    """
    x = cpp.Pipeline()
    def fn(pipeline,event):
        if isinstance(event, cpp.RunExceptionEvent):
            import traceback
            test.assertFalse(
                isinstance(event, cpp.RunExceptionEvent),
                "\n".join ([event.error.message] + traceback.format_tb(event.tb)))
    x.add_listener(fn)
    return x
         
class TestPipeline(unittest.TestCase):
    
    def test_00_00_init(self):
        x = cpp.Pipeline()
    def test_01_01_load_mat(self):
        '''Regression test of img-942, load a batch data pipeline with notes'''
        data = ('eJyVd3VQFN7bLwiKSAtIs7T00p1SUgLSXZIqtdTCLogoISUgHYuggkgo3Q1L'
                '7oJ0dzcrLLAs+/r7ft97Z9577z/3M2fmxJzz1HliHn1VEz3VJ2ySwiJs+qom'
                'Qq4er1wE2QxfOfq7evu9lmMzVDPXfibIpubn4ujv8oLN20uOzfzvrO8IZhOV'
                'ZRORlRMTlZOQYRMTERVh+/8GHr62PgUeHt5HZjy84NxhS3vvcRHKILjUPb1x'
                'SiAbF80h8Ux7DQGh92HTM1J36SmP3/lv5xvy2H8VyTAIwd4kwTkBRM36pVM6'
                'mxPaUhyWk423i7YINHz7FR7e7StghOltcx/8Ib2yAJVtNuNbryK2ji9Isw6x'
                'YMYR3Ob6w2v6QGZ/4jxKURHw2hX17yI6Ne8z89S5z5FWo3l2wzCzncCPyGeh'
                'oiLXGFKKj+N97ZSZ0DWX7Z174nra75Iuh6qpGyU+btRV7iKdYT84T5dXiprs'
                'smeKuWa3ZQHzxignrJbS2vkQ8QVEoTZMvFD01ylZndoO/cx7bGzgW9j3lXqF'
                'bm+GiRxz2N2j1jv1HLvsO9TsCl8XQsUvQwtc551D+XxyUmxE9yYOHd6WuR6W'
                'OU0+mwOfGc2utvVojRYrHQFqPb8fGGZ3ZqjIVsF0jkq5zEfLf+6z5ZK386xO'
                'c7T2Nu+vzTs+PnGU7lcLT1tMdz3RLdBQ5ceBeQdo/buPOOUDl6yrwlOWmdk5'
                '6lSsleDS9/LbIOy5za1SbUfGQJIlVDLtOh0CwOeLuvm2A5EGIn48WODUrigA'
                '5/t9PvV76bY/6W5njtv6iT3X1cG6hNqf8hfluAFH2WZekwZ/E7Gh969EnZin'
                'sWr72heTn39tAHoCxrdYaGEBsUkF1rx/mDYYU4Srxs/AbKdU/iO+azRSY49f'
                'sgzoJmkgRV/vdQlivgtUuodybxFEV3loXQP2DIwumvebiqyVTfziEOHa4Ehn'
                'IcZ75MqoRKsohT+j5ZzLNA3TKTaq5nqoSJe2ar0wK4ChUx4ls8OOxeuJICri'
                'MoNPEp8TGLidcz+irNm3n6dw7kDHTE/7AgAPd0gZXDG73KivLuNiy1abFGYJ'
                'NzIy3pMnogGvNvYncyI/Yb4Misf1ucU9NHmhlLdNJ3dLP/nkEGTUWAU6ebRh'
                'rWUKSVmrjmUjgXIp8AdAeNAurAaS8bTgNu3fuPvVKq/e/XrJ7PobNh1VA/X4'
                'HRaoAR2+UmzjuzzQhhCOxHFy1anKLxtJO2i6uatEgkviQuln7zEsijPUuAWv'
                'tX3YxdhD6Q2REcMJW7Woawuj8pBb12KnwQZTYjplUeE25WFxO/9LrsnewjOX'
                'JWiZV77Q3kIqDdKsyGY3iyiQFf0SQHv5+KHdKFwpAC79VZNCPqZHul7uakwv'
                'NvXQ68i0ZqNVmw+YEC74MEkO20F3sPZmd9e64sV8UumdevZdgR2jvE6uF0V9'
                'Njg7F//Kj51e5x/vToyglRyM5M3e3KJjtLGyeTqFbENQuvPXeg43i9yVCkE5'
                't7DJpA/DoLgZVenjXD4fIBf27uK7TAsGTHHLvq1phG4oMtgcGmi0XJSy19Kv'
                'lkl0aDshsIiyGjuW9bY3OT3VSxHYAc2tpq1RBob/gTKeIQpd1YAd0u6D5KZ3'
                'l3tAEswxUoRwVvnj1lt8PDwc7sBywGtchV5qkfeOjizew2IFFZck4zsjQl4e'
                'upSWMdxcDfHMoT76cPHkjN86zMGEicxgEwXtBSljrXW9Qy4NC0bu9uxj+63l'
                'xQANI6lU0PGcIzLwTN9AqT4FJSzLtL5cnoALY0E73ZEuDK0BRI+1ucCa8BNi'
                'jWX7arxX2ZXZVurFOr01PIy56bwC7tSXQjRXKesfd5SDrsWvT0jv0ukoNYL/'
                'LK747KaumfXBwBvl+v3WuhtUPT5k+EdUbo+LrKfA78PKg9s2C03adtqOQBD6'
                'RWKFaffsr3NCTHQ30CyAskoUhq+jVkaeoZysiwSjzslLXvJoDv8JoZzZC497'
                'souHtWTHMU5Ed+fwuzKv81nMsg881cpf1iwHhtybwC/ZlSshwPt9c809uk5S'
                'TFRE4RvTBivA+156XeT+1e/a2aqVVyxsEAFKvaBCeAABWteJ5Om/lGjSHoMP'
                'c7aSLH1Vypf5j7ZMvSOazK3fLbusMR2wrW3UgJZ8+ZmkqwLcjLNaAJrVTdVJ'
                'zjX35cNJfKSKgUWP95VezrP6SsZZXHL8eVL+QIXl7TSlyWffZ8SvHgLPBEn1'
                'N65j1E8UuYWGTp/opMD28N4yTuz7VOlv1MUh40KwI0ccN8IClewdr0FoJAL/'
                'HMOq+tPFP1XAK1vsjAwCIs/nXlIPUd744+r/+MG6/z2Dq0/cOOLHsDEt1sSU'
                'kgA5rHd/otd189Mr61yIZs5EvRVrbLcaXONZW+sWHzx6TOVut5sWIDxLtbLs'
                'Kmd3MpYzt8ck8yy5NaPwkOo0FHHqXAN+2LNBkxz4ynnD1rKuv22acqV6zSUT'
                'aEsTv1a7ABkYnJawn9GzkYBx7KHEhn+M/6HpPRQ/yqcvhxbKXKSmuTfodI7x'
                'm1bdjl9bqr0ZKq0c/Xb7JsruHZu+Zrb1iFpZ/eJ7sJH5j/qUILqLIzE0/VTo'
                'xgWgYOjyR4bE0tWRh/T5pzTzF+2DITgjXSsfPy6UCm4b5ZB+6XnfJ0k2lH7E'
                'Ffy08t4vjT/yWQ2nkOldAtu5I+LpzQn7C+x8TRgI8sLfZYM8xnpvqX+9lKFB'
                'A7HtMlJEIaNIz9JD/97Tn5WWLdzutpSuxC0LNOyluWivqWifcH4DgPuhM0wr'
                'XZPrD45JXK4zFH8yL3OqQczfhA+sMkAb6xP0IzO+uWC+H8geXeN+3d4/ISBX'
                'zkxqlxiXUyNnTnAI/zcELV7+rdc0Ac0V8dGUJxFIdxjv7+fFX53TDD0M5Y2+'
                '1I29JXTPFkwNztLYM9GWK3mD6zLgPlKT8OBBp0nkPeIrffZpHASBVlbmwfHe'
                'UOMQGHRE+HbH5RxsJT58OefmbH+AMIjFk1+IzEeJ5gR950bxJK0B1AKJkW79'
                '1nKwEUTVaqMCJCog3JQ5F3MwAxJs3F8SCwmIWCKgzKVIzInSjjejyHlRLmMq'
                'RVucuS2eptQNtnvr8/istsbV5vbBn2u2zEpmaaGMmwYO8mSu3aDk61jdLMse'
                'Umu1jCYNZX9TwiQqQfbN4hHDOKMkJxKPngOqrc9JUi+dI5ukUPZp7G9pucyT'
                'u9oykHIPFGuFImU0yOypCRI+dCWI5892XFYVOCeBCwMkAR9I4E1yPkOITJxq'
                'R/ijNZACYLAXToWIm3b45bswAWBkceFeTL2LjdrXT+cXOC3ggjfZLVP0Rq8/'
                'Qt0HfZfOvYJ92WPhKZM+jC6UUXzQG+s8crR61FHNi5ImY/rTSmpgEm2g1iY7'
                'FeMlTiDfbYmyPmvfu659tbw3E7+wpZBolZE+Rl1eRMSWqcVcHxtNdrBqul+s'
                'KnzaccZ4zL1uPdEgvatcCmMwvJ6Qs4XetCTV0NbWHII0H7Y0OUB80cnAO9ju'
                'j7erZUqe1JFOFwADNy+ovfbhNBaKzkNKQpYmtPbu2C2AhfaWkWrXzALknXO0'
                'R9QTEwG4nlVD6AueAfCza3ltE9yzSKRGW+UrP8A1JSEqYn1L0jcK+OezV/LT'
                'ZY8jgaGpctIgaWRqRZI8LOcczISqFh7V9WuNBdkVIhI8Tho9cTEgiEcJ/Auy'
                'HsOWuc2MCZ314/56OpumVCugJG3a7v3wgvjr3gI6mguhPz+cwQW5uBsiSWkD'
                'lDFLe1YU+q0g43iivvS94vNUO1rNHfFfZXONfl/scruL9norf8O6uMMrjHDc'
                'MiFmMegWKgTTeGf36+nPMWiKf10bqdstQoMIkynumiiqZw0cZdnbGoh0j3Ph'
                'dZf2L1hTf8LUpPNac+4Zenn+p/7TvMDxcu39O58h7zPQm6HA468jkWMl76pZ'
                'Ec9lhlYFE5crE3yFc69OUxPzKHcvf9jyOUkzUMsb3Yw4F/ecLDgFJ/psk+lG'
                'EShSgMMv2tdGzpZiTr5dQnbWw/ox50y48Sk+kX2aDrWG7ZotlES4Rp89lQUh'
                'ObhH3MP2yPnGFD9HmWaoFu6FPvHAlUELVQF5g7WRSDrpCF3fgCnsRuY5qKBT'
                'm2gMcSjlEFDcoH9g/jSqvCPoVdDAtnd6A9fncQ1S5GLMM5pl9vm9q/5P9z7O'
                'jEqO3gm+Mb/9cfOo2ktA9FYK/wewlzHd5kA3Ivry1M5MOebCm6U5Bn2r3gAU'
                'RC3xVOXd2VIG6Ot91PBZyRZYshxZDoxvV+zY0/2lQ9ly2KkU0hYFzCw5rgt6'
                'bzN0zP0znArHydWJkulblnNYV/eNm0n/9mMBJSMUaHkcbTnNLaSgcURk2a7c'
                'Ozkm5rDP25s9Z+TSzXSjCZj0bQgHxDBTjCGM0/c0pgIGnKAPy5e01pdKUH9f'
                'hQhXAdPo8zlfsQ7kDfGE2447Seve7PvlySO5DmhOqoVpO7aYevXnT6lUv7HS'
                'k1fhX+L9882Zy7pdhjRw5ARHTGFxQfHFitqDJpd23q5PHCH3WWhNSgZqdCu8'
                'jBdoY3fCvetHZqJKfs0elcvTBy2/BiAal/PyEpof4JW2AucwKfK/X3dOFun7'
                'wd/JR63qsArTrnCh86nqt/ZABNQTWklpqkycnDY3Mo2+KP3rrRxCyUqEpF0X'
                'HRA/zeum7ecJ89u2g2OEdD7hUvBDGvqT93ReuTC+k+usqv05dd/tgbdAwbbj'
                'lo0BXSjit30dw8mmtAwZD6YK386e2VYyWe/PI00/ivNNr32G2li9pRDeNXBM'
                'G60QehrTb9BAknkwJHe5n++JCw8IUPnpyKG7Ady7wvmOtzHlrFQql8q/+5NI'
                '0xs8CRulrM0l0JWdHMyIunJ6P1gszg2LVFEyUX6V5w5nnCLm36rO4jGtv7/u'
                'VJ47zigpGTQTDIS8yH7U6vN6JgiaN+RzntuDGWCFD4WTAcGasu0r178VH0xn'
                'EMI1YQ05D4GHcds7kDqzSgGyuVvkQlIPOvZHhHZA9l1c8KqrNXIxVtGgE2vg'
                'VXe9EPd3pTEUe95xYkPMg2kktVNmgYofDL4us35k+27lVdvb9RyBzo0QW/zw'
                '1udY4n8+MQvp2aVC2dfeY9ITOSN9QvwhnBIU/+CrYk0IZY/z03QuXqanE9ve'
                'j+p+8nkwfsi0zX3vmNlN0ZZ19UTqBZ1bfMLbVNi2IJEIr5ONen1M/vCawsA1'
                'xkJsXoy6mGKbPxcVWf1JEGE4lSMYq5Q+/10bupQTqoa1gRnl/yhBC8auiZNo'
                'mSvrQlhFCDA8iFeJEN0Fb5s0gGBJ2E+RMZViRG70NfjCKKxNvfUs0vnTVzrW'
                '7GmgryuItb+yB4vNG9WYJSokdwM2o82rrycOsZ5qeT8SkDXQvB5FHvNNJLlJ'
                'eKCPA3hfLrZZaHpjSXlcpZjPJhXcxLjDdSTayHOb+7IgtWHezzQ0qOBoke8I'
                'lltbbux2pIwznuJTfnzzBMSdCNFZsG2gP8fYF7SEG/131Z46JulUoSc6cRtO'
                'Mj/TCnb0FNcx4uDeJCPRGHlqaSU0+Pe0fX52jHs1IRNjZzsNmi8+Ompc9Er3'
                'tgoYsj9YjDIa3RJkbG5b/mndtDnzzQ6UuJkqtfyLSA0YvP+72CgqZriAJzJ7'
                'S0iAmNU+rmue3tsFtJfnfAy/m0FW+MEjDJrr4yd3YABeeJe/rajltNQ9wzsz'
                '7jHmCA/PwXF5sUR2Z/c2lpR3V0K9+WP3K557Vwe2Kz6TkWW5KM49MlqifYiq'
                'MNwl6iMm+gt0248HR2OyMI7TXIr12ynr2qbupyPU5P8BOkRGRqYhPLDy6Fz7'
                'IfJKxC493qieuhzcBUXk/WuFGVWStyI0RCcMPHmFFKufibM1KFXpVbc/ytGU'
                'aBN9VjkwzCs8XkovSo+PEebHEiUdjmAdDk5Y3T9CJCxr+l+6O3nFNI50uTjG'
                'gHx8fDZlW1ck+C53fk7+vIN+F/kXmJHvey3PNuQqU3tFdQfbHjP/xQLaCyhr'
                'mgegakRsX9XJnz8Oa5J/nO88lum2z6Xf6uCKMVqZW9+Raxi77FEuIz+YcE6a'
                '2rNd+YvbxddcxXsUPOBjCoWgMjbSUCogDjTGlc80pNsHqvY6xvvHGI0AGhwO'
                'LeBj8FfxvSb5f/NbxsiDLhHSN9Y4SiIxpF1L6FgOPMTI+/rBbbyoqnRUW3HS'
                'xl6pv0ad2FwFC/G7lM4MCXbdOvJBl9xdjnHVb2qBz3fFMQKZeR2762bM09Qy'
                'w3kmENfjcd4FrymijDNCcpmVNwQVkCXnHmRofPqjBer/7MllwEeTvcrkJAsS'
                '47Loha8te37v1m8PW6ngV8OJXnCR++OXWTP7xV6KgbSnj59APOHFL33CIMIi'
                'cUcZbKs3knkGphCDmGNa1Birval9/9ZqNmB+SXm2xaRdq2J4vA5EODtZ2XR2'
                '8v4yjD1sKZ8fO9MUVHpTkg2qCvCctzM3pZu8yqrI2FWqclHOKo17vUzVtN0s'
                'wKxqWMyz/wODg9MmFpzJ7S4hv8labylvLv13DXj9HxtF2Wto0LzQOrnPym5+'
                'WhRP3MQeEk733YKoQtlf6r005ZMCZMo39poc/Mz6UjOXPWxxfAtLaLNnwPo0'
                'Li71TY6/jK09oc1qrdxrXXO7PevqCgTRNClD+GxsZVirfBP2PLljGoN8T/1n'
                '2iMCLfcdJ1d8c6GbiHVmP7a4+bzNKevHdy15wfIDxUH3bNTefim033KN0AMj'
                '+QiGimGW9dvoT67u1u44xd5kUX8pTz2/NDsO+tUyu3wDVD1C576BOiRTn9j0'
                'Jxs3Evyjz6TRg3dslG8c4xZjWB8l/dTauQ9KK4hhDZR+1BSuozkVaaBqietg'
                'ewvgQ1gUfnu6eClmmT2st+VVc+qPknUKXGuE982ev9+uyQmClb6ew3nvrMsD'
                'yaO7FxdjtxOD0Um3Somhnas+f5S9lB/z2yhble/Dt54rHx9ZSyti+lYRm+rv'
                'MZG4O+TRzTnUyFPhNZ/pHYoLFi3oqk9oI9U/4k0b/eOSq7zGLwhmqRTeavWb'
                'Lgg6v52nbG0XIpyKvZQxx5KwE14rEk6V0VoLDL0cCgVoAawB8kOOr8v6VtsO'
                'Jfxm40gwqP40s/pp4/Vsaxz973266O7V20WPjaUR7JNvIOUJaIDvOZT5hu88'
                'rlto9jtEKemrHtI55pQ10L25mun5Hl1GnfFq6CZOwb69Rc1Xhtkr6xknHIr3'
                'l4iPzEJl8+N//WJK90GkCM37VV6rnogd8qZ3wp2f50+o1DuJvi62x2rGr5UU'
                '3bslyfBhB3BRkp/2msXuZexF7/WXNT0RP97Mov0PsWgBQTqb3CWYpbAkyuS8'
                '6rbmeFdeuflXsvg2fNWHvGnBL+iKKg4UttPINGhWcbq6BPX1arY5sfOVWQjf'
                '0OdHgIsw/TmHoSONf0Uk/0c4ZnJ0nPW/0iE7id+IkL5f5efP/8Ip85az4YMD'
                'n2KJhk84U4yhQnBaFyBY4c4V+W68KPRDjd4PjHhaYLniF72QpH90hHkhbFB+'
                'ft7ibpKO1i2obd9e0YMfcdzYuH+ZRP+PCb3p9T9Z8v6fLK1jDBcK/lAosEZA'
                'okk5xPq+PBvfv9UvWvQ0EI1bWv+/KFYI1rtWYBr9vcXcJG5ncgPQ5+9mYc1k'
                '/5u9eWrovzl6qJMYj42UoIOfty23F1ts/PloZf0OO02IIkff7mJOOlp3oLwR'
                '5b+pDPdl/n+N8Aayf5Mg8j9tKjzH/zWRxijxZ3Z2/alnBN9KOmrGycfr0vWd'
                '9r4cdRBvcVfqsyVwzueZPbshvpFlvhr/SV0/oiceZBwR8Vwydjmu/XbjCq5r'
                'w5emM/R1F/nc3NoW8KGIL7G01Nz09on5zgPbzwL+rotemvVfdl9gNTWnDbVE'
                'OqA7xmLxRpRgSpzu6aOm/3XrfcUobdMTtPL+aOR3xG//32Hpp0PXbgyIMmxH'
                'nB42qsr2fndmih3lnYB7PUxRHbkfHYCzvCLbebys3L1XsjmMbuJJ48XBPbG6'
                'T57k1RdCdC5cwKunIhB8/vwK3r3TFmkfKQmEJutjR1IRdB6hr6AOqmfV/VAe'
                'fKi2sd5/9+QFV/mEO1hbMAXkscLsmatGEG0oFe/8IP4cD8We90S+5sdRzWQe'
                '+DKUMRxYD1CMCR5l6jX5w8uSjDJY8dti5kNr7x19OdqVEY5XxhtWN+STjpFR'
                'F1zh3JYJKF4o+hU6KJ0AMl6XRl12ranAHeupdTVC6fqugirpjCGvy7F1Hqyi'
                'PtobI5nai6cJAzYpkOWO5d+X1mCIH75ED6fYhiMlcXPqHItL0vZIbES0BOvg'
                'Otn9BOi2Rf7xE+W+e7cVfcsjecUZookKovpQRnUY/MRsjdZY4nTJAcrsvkBg'
                'vOB0deqgOVM8CBtQR8VvZ/fZD603f+pBtMkIybQvfr8ZfAvsgDB3T978bVUr'
                '/A2u3O7dFp7tw+xw3xoeneeh6peTvkF+Fp0GnV1yACgHZR3XJt8pbhH4ZJG9'
                'O/7kX9wmOi2RP88YNN9zLgLfzuwkDClgJtlq5KAh385sbmvpVBf+mKi4664L'
                'r7zaWiK1C4Q5CGA2B3Va7TQ2NWvbmmwsy01Z2TEDZzpXp+OY1NMmr2Unl8Qa'
                'WwG+5QtwaWBv1aFVo5P1mbdloD4N/6aVs8eZwmxYOdOc7XNsyDxsA9q+eruv'
                'tXt9nW6Gs969Bp999g/If9ocsBulBoy5dsc670/u26ES6hMt2NomDTuSkxi6'
                'qB7yMPSw+6iQynAobnxANt5/EiXVI0B9l9QXQBnblSyD/5Bfyh+7/ezaxt6b'
                'Hl6GXjT4J8wPTTu9xlUo9SnZGmgS2YQ5hYWeM7L3sJR+i6ExLb//lLn5Fxzx'
                '4bxi01Z6oHytJ5kT1plssCIu47dBnWk5Xz6bbmEkG/hVX+zixdbIcE6Wdavd'
                'Mfmi0mL0uTQMU9tib0UUy8eWLmXuRAy23OgrLD+X2vEaa12s6i7+NZ8bpotr'
                'wHljmc6r68KWqsLFR91a6fC90oP0pdIYHUm+5jjiEzmxV4sRfHjxOBGmIkxX'
                'sqJe8lDoKRmRvBZSZYc+sC+V/Q0a09p9XNwBHKL+zllWpk3C41vCF8AwTEx4'
                'R9WR6MbTqt9/695tsjkgiYeegSXOiRjk2CmXuv28eCU+4qkRYpfOIjU2gihb'
                '++WVpOKpFIne17HA4Px6b4G938e2wygPrRaMaotgez195MPE53o7D2oqPiZj'
                'BOhf8j5dVZGLIbNq3cWjk412kMBqOVs+nab84Cs41YNtIkBH4kkPhYbwNDyv'
                'DWL7WUE59asVJnBj34fMYLmVvoMTNBAZMZQfdfliaIZ0ZQnMJwq0yB9WT5zn'
                '9MijiI4FXbrijad4FsRC5m+pt75dPn2fXVikI9kwfTi/jJMFot8GW4Rx04IG'
                'KG9NtH0jKDJmZ8kOfh6AZUtdp57K+JbAs556sIocl2yXCBh9VmO829Op/tsY'
                '/7voTYqFgji/Utz60IcG4Pq6L9HSIR5wyFtU/PECQl+elN580qJu6wFa6Tze'
                'aiosrNwyaZzcZh2PbijVTweqyvI6hfBi7VMVP5Qjp8IrKGR8ycNxi62wy4Kp'
                'ph5+6n83cavGVwKH+eXK0hbYk0MYZrH0g4faYM5syoOT+fnS5xdmY/I26qXx'
                'Y6iouC23povqluz9gPKU6ljAWMNYjS9EI5q2DfmyD/WZMx+uqiA4LWTIvq2o'
                '5Ohg8lP9UZZPjSjiED21MGR7pU+gIGhOUxqSXMfG3SsoG2kcT+j4Chirzj/5'
                'QjLVt+yntvsIX+lgMwPD7MbhU02piHSKcpBblPnabBPo1ApnheO/1hLCNkco'
                '7St67i67b+xJLI94h1/d4HsGhvD8F7MQ7RE=')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(zlib.decompress(base64.b64decode(data))))
        module = pipeline.modules()[0]
        self.assertEqual(len(module.notes), 1)
        self.assertEqual(
            module.notes[0], 
            """Excluding "_E12f03d" since it has an incomplete set of channels (and is the only one as such).""")
    
    def test_06_01_run_pipeline(self):
        x = exploding_pipeline(self)
        module = InjectImage('OneCell',image_with_one_cell())
        module.set_module_num(1)
        x.add_module(module)
        x.run()
        
    def test_06_02_memory(self):
        '''Run a pipeline and check for memory leaks'''
        from contrib.objgraph import get_objs
        
        np.random.seed(62)
        #
        # Get a size that's unlikely to be found elsewhere
        #
        size = (np.random.uniform(size=2) * 300 + 100).astype(int)
        x = exploding_pipeline(self)
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:10253

IdentifyPrimaryObjects:[module_num:1|svn_version:\'10244\'|variable_revision_number:7|show_window:True|notes:\x5B\x5D]
    Select the input image:OneCell
    Name the primary objects to be identified:Nuclei
    Typical diameter of objects, in pixel units (Min,Max):10,100
    Discard objects outside the diameter range?:No
    Try to merge too small objects with nearby larger objects?:No
    Discard objects touching the border of the image?:No
    Select the thresholding method:Otsu Global
    Threshold correction factor:1
    Lower and upper bounds on threshold:0.000000,1.000000
    Approximate fraction of image covered by objects?:0.01
    Method to distinguish clumped objects:Intensity
    Method to draw dividing lines between clumped objects:Intensity
    Size of smoothing filter:10
    Suppress local maxima that are closer than this minimum allowed distance:7
    Speed up by using lower-resolution image to find local maxima?:Yes
    Name the outline image:PrimaryOutlines
    Fill holes in identified objects?:Yes
    Automatically calculate size of smoothing filter?:Yes
    Automatically calculate minimum allowed distance between local maxima?:Yes
    Manual threshold:0.0
    Select binary image:None
    Retain outlines of the identified objects?:No
    Automatically calculate the threshold using the Otsu method?:Yes
    Enter Laplacian of Gaussian threshold:0.5
    Two-class or three-class thresholding?:Two classes
    Minimize the weighted variance or the entropy?:Weighted variance
    Assign pixels in the middle intensity class to the foreground or the background?:Foreground
    Automatically calculate the size of objects for the Laplacian of Gaussian filter?:Yes
    Enter LoG filter diameter:5
    Handling of objects if excessive number of objects identified:Continue
    Maximum number of objects:500
    Select the measurement to threshold with:None

MeasureObjectIntensity:[module_num:2|svn_version:\'10087\'|variable_revision_number:3|show_window:True|notes:\x5B\x5D]
    Hidden:1
    Select an image to measure:OneCell
    Select objects to measure:Nuclei
        """
        x.load(cStringIO.StringIO(data))
        module = InjectImage('OneCell',image_with_one_cell(size),
                             release_image = True)
        module.set_module_num(1)
        x.add_module(module)
        for m in x.run_with_yield(run_in_background = False):
            pass
        self.assertTrue(isinstance(m, cpmeas.Measurements))
        self.assertEqual(m.image_set_count, 1)
        del m
        for obj in get_objs():
            if isinstance(obj, np.ndarray) and obj.ndim > 1:
                self.assertTrue(tuple(obj.shape[:2]) != tuple(size))
        
    def test_07_01_find_external_input_images(self):
        '''Check find_external_input_images'''
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9870

InputExternal:[module_num:1|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Give this image a name:Hi
    Give this image a name:Ho

OutputExternal:[module_num:2|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select an image a name to export:Hi
 """
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(data))
        external_inputs = pipeline.find_external_input_images()
        external_inputs.sort()
        self.assertEqual(len(external_inputs), 2)
        self.assertEqual(external_inputs[0], "Hi")
        self.assertEqual(external_inputs[1], "Ho")
        
    def test_07_02_find_external_output_images(self):
        '''Check find_external_input_images'''
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9870

InputExternal:[module_num:1|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Give this image a name:Hi

OutputExternal:[module_num:2|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select an image a name to export:Hi
    Select an image a name to export:Ho
 """
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(data))
        external_outputs = pipeline.find_external_output_images()
##        self.assertEqual(len(external_outputs), 2)
        external_outputs.sort()
        self.assertEqual(external_outputs[0], "Hi")
        self.assertEqual(external_outputs[1], "Ho")
        
    def test_07_03_run_external(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:1
SVNRevision:9870

InputExternal:[module_num:1|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Give this image a name:Hi
    Give this image a name:Ho

OutputExternal:[module_num:2|svn_version:\'9859\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
    Select an image a name to export:Hi
    Select an image a name to export:Ho
 """
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(cStringIO.StringIO(data))
        np.random.seed(73)
        d = dict(Hi = np.random.uniform(size=(20,10)),
                 Ho = np.random.uniform(size=(20,10)))
        d_out = pipeline.run_external(d)
        for key in d.keys():
            self.assertTrue(d_out.has_key(key))
            np.testing.assert_array_almost_equal(d[key],d_out[key])
        
    def test_09_01_get_measurement_columns(self):
        '''Test the get_measurement_columns method'''
        x = cpp.Pipeline()
        module = MyClassForTest0801()
        module.module_num = 1
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 6)
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'Group_Number' and
                             column[2] == cpmeas.COLTYPE_INTEGER
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'Group_Index' and
                             column[2] == cpmeas.COLTYPE_INTEGER
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'ModuleError_01MyClassForTest0801'
                             for column in columns]))
        self.assertTrue(any([column[0] == 'Image' and 
                             column[1] == 'ExecutionTime_01MyClassForTest0801'
                             for column in columns]))
        self.assertTrue(any([column[0] == cpmeas.EXPERIMENT and
                             column[1] == cpp.M_PIPELINE
                             for column in columns]))

        self.assertTrue(any([column[1] == "foo" for column in columns]))
        module.my_variable.value = "bar"
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 6)
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        module = MyClassForTest0801()
        module.module_num = 2
        module.my_variable.value = "foo"
        x.add_module(module)
        columns = x.get_measurement_columns()
        self.assertEqual(len(columns), 9)
        self.assertTrue(any([column[1] == "foo" for column in columns]))
        self.assertTrue(any([column[1] == "bar" for column in columns]))
        columns = x.get_measurement_columns(module)
        self.assertEqual(len(columns), 6)
        self.assertTrue(any([column[1] == "bar" for column in columns]))
    
    def test_10_01_all_groups(self):
        '''Test running a pipeline on all groups'''
        pipeline = exploding_pipeline(self)
        expects = ['PrepareRun',0]
        keys = ('foo','bar')
        groupings = (({'foo':'foo-A','bar':'bar-A'},(1,3)),
                     ({'foo':'foo-B','bar':'bar-B'},(2,4)))
        def prepare_run(workspace):
            image_set_list = workspace.image_set_list
            self.assertEqual(expects[0], 'PrepareRun')
            for i in range(4):
                image_set_list.get_image_set(i)
            expects[0], expects[1] = ('PrepareGroup', 0)
            return True
        def prepare_group(pipeline, image_set_list, grouping, image_numbers):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PrepareGroup')
            for image_number in image_numbers:
                i = image_number-1
                image = cpi.Image(np.ones((10,10)) / (i+1))
                image_set = image_set_list.get_image_set(i)
                image_set.add('image', image)
            for key in keys:
                self.assertTrue(grouping.has_key(key))
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            if expects_grouping == 0:
                expects[0], expects[1] = ('Run', 1)
            else:
                expects[0], expects[1] = ('Run', 2)
            return True
        def run(workspace):
            expects_state, expects_image_number = expects
            image_number = workspace.measurements.image_set_number
            self.assertEqual(expects_state, 'Run')
            self.assertEqual(expects_image_number, image_number)
            image = workspace.image_set.get_image('image')
            self.assertTrue(np.all(image.pixel_data == 1.0 / image_number))
            if image_number == 1:
                expects[0],expects[1] = ('Run', 3)
            elif image_number == 2:
                expects[0],expects[1] = ('Run', 4)
            elif image_number == 3:
                expects[0],expects[1] = ('PostGroup', 0)
            else:
                expects[0],expects[1] = ('PostGroup', 1)
            workspace.measurements.add_image_measurement("mymeasurement",image_number)
        def post_group(workspace, grouping):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PostGroup')
            for key in keys:
                self.assertTrue(grouping.has_key(key))
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            if expects_grouping == 0:
                expects[0],expects[1] = ('PrepareGroup', 1)
            else:
                expects[0],expects[1] = ('PostRun', 0)
        def post_run(workspace):
            self.assertEqual(expects[0], 'PostRun')
            expects[0],expects[1] = ('Done', 0)
            
        def get_measurement_columns(pipeline):
            return [(cpmeas.IMAGE, "mymeasurement", 
                     cpmeas.COLTYPE_INTEGER)]
        
        module = GroupModule()
        module.setup((keys,groupings), prepare_run, prepare_group,
                     run, post_group, post_run, get_measurement_columns)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run()
        self.assertEqual(expects[0], 'Done')
        image_numbers = measurements.get_all_measurements("Image","mymeasurement")
        self.assertEqual(len(image_numbers), 4)
        self.assertTrue(np.all(image_numbers == np.array([1,3,2,4])))
        group_numbers = measurements.get_all_measurements("Image","Group_Number")
        self.assertTrue(np.all(group_numbers == np.array([1,1,2,2])))
        group_indexes = measurements.get_all_measurements("Image","Group_Index")
        self.assertTrue(np.all(group_indexes == np.array([1,2,1,2])))
         
    def test_10_02_one_group(self):
        '''Test running a pipeline on one group'''
        pipeline = exploding_pipeline(self)
        expects = ['PrepareRun',0]
        keys = ('foo','bar')
        groupings = (({'foo':'foo-A','bar':'bar-A'},(1,4)),
                     ({'foo':'foo-B','bar':'bar-B'},(2,5)),
                     ({'foo':'foo-C','bar':'bar-C'},(3,6)))
        def prepare_run(workspace):
            self.assertEqual(expects[0], 'PrepareRun')
            for i in range(6):
                workspace.image_set_list.get_image_set(i)
            expects[0], expects[1] = ('PrepareGroup', 1)
            return True
        def prepare_group(pipeline, image_set_list, grouping,*args):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PrepareGroup')
            for i in range(6):
                image = cpi.Image(np.ones((10,10)) / (i+1))
                image_set = image_set_list.get_image_set(i)
                image_set.add('image', image)
            for key in keys:
                self.assertTrue(grouping.has_key(key))
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            self.assertEqual(expects_grouping, 1)
            expects[0], expects[1] = ('Run', 2)
            return True
        
        def run(workspace):
            expects_state, expects_image_number = expects
            image_number = workspace.measurements.image_set_number
            self.assertEqual(expects_state, 'Run')
            self.assertEqual(expects_image_number, image_number)
            image = workspace.image_set.get_image('image')
            self.assertTrue(np.all(image.pixel_data == 1.0 / image_number))
            if image_number == 2:
                expects[0],expects[1] = ('Run', 5)
            elif image_number == 5:
                expects[0],expects[1] = ('PostGroup', 1)
            workspace.measurements.add_image_measurement("mymeasurement",image_number)

        def post_group(workspace, grouping):
            expects_state, expects_grouping = expects
            self.assertEqual(expects_state, 'PostGroup')
            for key in keys:
                self.assertTrue(grouping.has_key(key))
                value = groupings[expects_grouping][0][key]
                self.assertEqual(grouping[key], value)
            expects[0],expects[1] = ('PostRun', 0)
        def post_run(workspace):
            self.assertEqual(expects[0], 'PostRun')
            expects[0],expects[1] = ('Done', 0)
        def get_measurement_columns(pipeline):
            return [(cpmeas.IMAGE, "mymeasurement", 
                     cpmeas.COLTYPE_INTEGER)]
        
        module = GroupModule()
        module.setup((keys,groupings), prepare_run, prepare_group,
                     run, post_group, post_run, get_measurement_columns)
        module.module_num = 1
        pipeline.add_module(module)
        measurements = pipeline.run(grouping = {'foo':'foo-B', 'bar':'bar-B'})
        self.assertEqual(expects[0], 'Done')
        image_numbers = measurements.get_all_measurements("Image","mymeasurement")
        self.assertEqual(len(image_numbers), 2)
        self.assertTrue(np.all(image_numbers == np.array([2,5])))
    
    def test_11_01_catch_operational_error(self):
        '''Make sure that a pipeline can catch an operational error
        
        This is a regression test of IMG-277
        '''
        module = MyClassForTest1101()
        module.module_num = 1
        pipeline = cpp.Pipeline()
        pipeline.add_module(module)
        should_be_true = [False]
        def callback(caller, event):
            if isinstance(event, cpp.RunExceptionEvent):
                should_be_true[0] = True
        pipeline.add_listener(callback)
        pipeline.run()
        self.assertTrue(should_be_true[0])
        
    def test_12_01_img_286(self):
        '''Regression test for img-286: module name in class'''
        cellprofiler.modules.fill_modules()
        success = True
        all_keys = list(cellprofiler.modules.all_modules.keys())
        all_keys.sort()
        for k in all_keys:
            v = cellprofiler.modules.all_modules[k]
            try:
                v.module_name
            except:
                print "%s needs to define module_name as a class variable"%k
                success = False
        self.assertTrue(success)
        
    def test_13_01_save_pipeline(self):
        pipeline = cpp.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.save(fd)
        fd.seek(0)
        
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[-1]
        for setting_in, setting_out in zip(module.settings(),
                                           module_out.settings()):
            self.assertEqual(setting_in.value, setting_out.value)
            
    def test_13_02_save_measurements(self):
        pipeline = cpp.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        measurements = cpmeas.Measurements()
        my_measurement = [np.random.uniform(size=np.random.randint(3,25))
                          for i in range(20)]
        my_image_measurement = [np.random.uniform() for i in range(20)]
        my_experiment_measurement = np.random.uniform()
        measurements.add_experiment_measurement("expt", my_experiment_measurement)
        for i in range(20):
            if i > 0:
                measurements.next_image_set()
            measurements.add_measurement("Foo","Bar", my_measurement[i])
            measurements.add_image_measurement(
                "img", my_image_measurement[i])
        fd = cStringIO.StringIO()
        pipeline.save_measurements(fd, measurements)
        fd.seek(0)
        measurements = cpmeas.load_measurements(fd)
        my_measurement_out = measurements.get_all_measurements("Foo","Bar")
        self.assertEqual(len(my_measurement), len(my_measurement_out))
        for m_in, m_out in zip(my_measurement, my_measurement_out):
            self.assertEqual(len(m_in), len(m_out))
            self.assertTrue(np.all(m_in == m_out))
        my_image_measurement_out = measurements.get_all_measurements(
            "Image", "img")
        self.assertEqual(len(my_image_measurement),len(my_image_measurement_out))
        for m_in, m_out in zip(my_image_measurement, my_image_measurement_out):
            self.assertTrue(m_in == m_out)
        my_experiment_measurement_out = \
            measurements.get_experiment_measurement("expt")
        self.assertEqual(my_experiment_measurement, my_experiment_measurement_out)
            
        fd.seek(0)
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[-1]
        for setting_in, setting_out in zip(module.settings(),
                                           module_out.settings()):
            self.assertEqual(setting_in.value, setting_out.value)
            
    def test_13_03_save_long_measurements(self):
        pipeline = cpp.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        measurements = cpmeas.Measurements()
        # m2 and m3 should go into panic mode because they differ by a cap
        m1_name = "dalkzfsrqoiualkjfrqealkjfqroupifaaalfdskquyalkhfaafdsafdsqteqteqtew"
        m2_name = "lkjxKJDSALKJDSAWQOIULKJFASOIUQELKJFAOIUQRLKFDSAOIURQLKFDSAQOIRALFAJ" 
        m3_name = "druxKJDSALKJDSAWQOIULKJFASOIUQELKJFAOIUQRLKFDSAOIURQLKFDSAQOIRALFAJ" 
        my_measurement = [np.random.uniform(size=np.random.randint(3,25))
                          for i in range(20)]
        my_other_measurement = [np.random.uniform(size=my_measurement[i].size)
                                            for i in range(20)]
        my_final_measurement = [np.random.uniform(size=my_measurement[i].size)
                                for i in range(20)]
        measurements.add_all_measurements("Foo",m1_name, my_measurement)
        measurements.add_all_measurements("Foo",m2_name, my_other_measurement)
        measurements.add_all_measurements("Foo",m3_name, my_final_measurement)
        fd = cStringIO.StringIO()
        pipeline.save_measurements(fd, measurements)
        fd.seek(0)
        measurements = cpmeas.load_measurements(fd)
        reverse_mapping = cpp.map_feature_names([m1_name, m2_name, m3_name])
        mapping = {}
        for key in reverse_mapping.keys():
            mapping[reverse_mapping[key]] = key
        for name, expected in ((m1_name, my_measurement),
                               (m2_name, my_other_measurement),
                               (m3_name, my_final_measurement)):
            map_name = mapping[name]
            my_measurement_out = measurements.get_all_measurements("Foo",map_name)
            for m_in, m_out in zip(expected, my_measurement_out):
                self.assertEqual(len(m_in), len(m_out))
                self.assertTrue(np.all(m_in == m_out))
                
    def test_13_04_pipeline_measurement(self):
        pipeline = cpp.Pipeline()
        cellprofiler.modules.fill_modules()
        module = cellprofiler.modules.instantiate_module("Align")
        module.module_num = 1
        pipeline.add_module(module)
        m = cpmeas.Measurements()
        image_set_list = cpi.ImageSetList()
        self.assertTrue(pipeline.prepare_run(cpw.Workspace(
            pipeline, module, None, None, m, image_set_list)))
        pipeline_text = m.get_experiment_measurement(cpp.M_PIPELINE)
        pipeline_text = pipeline_text.encode("us-ascii")
        pipeline = cpp.Pipeline()
        pipeline.loadtxt(cStringIO.StringIO(pipeline_text))
        self.assertEqual(len(pipeline.modules()), 1)
        module_out = pipeline.modules()[0]
        self.assertTrue(isinstance(module_out, module.__class__))
        self.assertEqual(len(module_out.settings()), len(module.settings()))
        for m1setting, m2setting in zip(module.settings(), module_out.settings()):
            self.assertTrue(isinstance(m1setting, cps.Setting))
            self.assertTrue(isinstance(m2setting, cps.Setting))
            self.assertEqual(m1setting.value, m2setting.value)
                
    def test_14_01_unicode_save(self):
        pipeline = cpp.Pipeline()
        module = MyClassForTest0801()
        module.my_variable.value = u"\\\u2211"
        module.module_num = 1
        module.notes = u"\u03B1\\\u03B2"
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.savetxt(fd)
        result = fd.getvalue()
        lines = result.split("\n")
        self.assertEqual(len(lines), 7)
        text, value = lines[-2].split(":")
        #
        # unicode encoding: 
        #     backslash: \\
        #     unicode character: \u
        #
        # escape encoding:
        #     backslash * 2: \\\\
        #     unicode character: \\
        #
        # result = \\\\\\u2211
        self.assertEqual(value, r"\\\\\\u2211")
        mline = lines[4]
        idx0 = mline.find("notes:")
        mline = mline[(idx0+6):]
        idx1 = mline.find("|")
        value = eval(mline[:idx1].decode('string_escape'))
        self.assertEqual(value, module.notes)
        
    def test_14_02_unicode_save_and_load(self):
        #
        # Put "MyClassForTest0801" into the module list
        #
        cellprofiler.modules.fill_modules()
        cellprofiler.modules.all_modules[MyClassForTest0801.module_name] = \
                    MyClassForTest0801
        #
        # Continue with test
        #
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        module = MyClassForTest0801()
        module.my_variable.value = u"\\\u2211"
        module.module_num = 1
        module.notes = u"\u03B1\\\u03B2"
        pipeline.add_module(module)
        fd = cStringIO.StringIO()
        pipeline.savetxt(fd)
        fd.seek(0)
        pipeline.loadtxt(fd)
        self.assertEqual(len(pipeline.modules()), 1)
        result_module = pipeline.modules()[0]
        self.assertTrue(isinstance(result_module, MyClassForTest0801))
        self.assertEqual(module.notes, result_module.notes)
        self.assertEqual(module.my_variable.value, result_module.my_variable.value)

class MyClassForTest0801(cpm.CPModule):
    def create_settings(self):
        self.my_variable = cps.Text('','')
    def settings(self):
        return [self.my_variable]
    module_name = "MyClassForTest0801"
    variable_revision_number = 1
    
    def module_class(self):
        return "cellprofiler.tests.Test_Pipeline.MyClassForTest0801"
    
    def get_measurement_columns(self, pipeline):
        return [(cpmeas.IMAGE,
                 self.my_variable.value,
                 "varchar(255)")]

class MyClassForTest1101(cpm.CPModule):
    def create_settings(self):
        self.my_variable = cps.Text('','')
    def settings(self):
        return [self.my_variable]
    module_name = "MyClassForTest1101"
    variable_revision_number = 1
    
    def module_class(self):
        return "cellprofiler.tests.Test_Pipeline.MyClassForTest1101"

    def prepare_run(self, workspace, *args):
        image_set = workspace.image_set_list.get_image_set(0)
        return True
        
    def prepare_group(self, pipeline, image_set_list, *args):
        image_set = image_set_list.get_image_set(0)
        image = cpi.Image(np.zeros((5,5)))
        image_set.add("dummy", image)
        return True
    
    def run(self, *args):
        import MySQLdb
        raise MySQLdb.OperationalError("Bogus error")

class GroupModule(cpm.CPModule):
    module_name = "Group"
    variable_revision_number = 1
    def setup(self, groupings, 
                 prepare_run_callback = None,
                 prepare_group_callback = None,
                 run_callback = None,
                 post_group_callback = None,
                 post_run_callback = None,
                 get_measurement_columns_callback = None):
        self.prepare_run_callback = prepare_run_callback
        self.prepare_group_callback = prepare_group_callback
        self.run_callback = run_callback
        self.post_group_callback = post_group_callback
        self.post_run_callback = post_run_callback
        self.groupings = groupings
        self.get_measurement_columns_callback = get_measurement_columns_callback
    def settings(self):
        return []
    def get_groupings(self, image_set_list):
        return self.groupings
    def prepare_run(self, *args):
        if self.prepare_run_callback is not None:
            return self.prepare_run_callback(*args)
        return True
    def prepare_group(self, *args):
        if self.prepare_group_callback is not None:
            return self.prepare_group_callback(*args)
        return True
    def run(self, *args):
        if self.run_callback is not None:
            return self.run_callback(*args)
    def post_run(self, *args):
        if self.post_run_callback is not None:
            return self.post_run_callback(*args)
    def post_group(self, *args):
        if self.post_group_callback is not None:
            return self.post_group_callback(*args)
    def get_measurement_columns(self, *args):
        if self.get_measurement_columns_callback is not None:
            return self.get_measurement_columns_callback(*args)
        return []

if __name__ == "__main__":
    unittest.main()
