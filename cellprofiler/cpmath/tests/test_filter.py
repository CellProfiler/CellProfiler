'''test_filter - test the filter module

CellProfiler is distributed under the GNU General Public License,
but this file is licensed under the more permissive BSD license.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import base64
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, convolve
import unittest

import cellprofiler.cpmath.filter as F

'''Perform line-integration per-column of the image'''
VERTICAL = 'vertical'
'''Perform line-integration per-row of the image'''
HORIZONTAL = 'horizontal'
'''Perform line-integration along diagonals from top left to bottom right'''
DIAGONAL = 'diagonal'
'''Perform line-integration along diagonals from top right to bottom left'''
ANTI_DIAGONAL = 'anti-diagonal'

class TestStretch(unittest.TestCase):
    def test_00_00_empty(self):
        result = F.stretch(np.zeros((0,)))
        self.assertEqual(len(result), 0)
    
    def test_00_01_empty_plus_mask(self):
        result = F.stretch(np.zeros((0,)), np.zeros((0,),bool))
        self.assertEqual(len(result), 0)
    
    def test_00_02_zeros(self):
        result = F.stretch(np.zeros((10,10)))
        self.assertTrue(np.all(result == 0))
    
    def test_00_03_zeros_plus_mask(self):
        result = F.stretch(np.zeros((10,10)), np.ones((10,10),bool))
        self.assertTrue(np.all(result == 0))
    
    def test_00_04_half(self):
        result = F.stretch(np.ones((10,10))*.5)
        self.assertTrue(np.all(result == .5))
    
    def test_00_05_half_plus_mask(self):
        result = F.stretch(np.ones((10,10))*.5, np.ones((10,10),bool))
        self.assertTrue(np.all(result == .5))

    def test_01_01_rescale(self):
        np.random.seed(0)
        image = np.random.uniform(-2, 2, size=(10,10))
        image[0,0] = -2
        image[9,9] = 2
        expected = (image + 2.0)/4.0
        result = F.stretch(image)
        self.assertTrue(np.all(result == expected))
    
    def test_01_02_rescale_plus_mask(self):
        np.random.seed(0)
        image = np.random.uniform(-2, 2, size=(10,10))
        mask = np.zeros((10,10), bool)
        mask[1:9,1:9] = True
        image[0,0] = -4
        image[9,9] = 4
        image[1,1] = -2
        image[8,8] = 2
        expected = (image[1:9,1:9] + 2.0)/4.0
        result = F.stretch(image, mask)
        self.assertTrue(np.all(result[1:9,1:9] == expected))

class TestMedianFilter(unittest.TestCase):
    def test_00_00_zeros(self):
        '''The median filter on an array of all zeros should be zero'''
        result = F.median_filter(np.zeros((10,10)), np.ones((10,10),bool), 3)
        self.assertTrue(np.all(result == 0))
        
    def test_00_01_all_masked(self):
        '''Test a completely masked image
        
        Regression test of IMG-1029'''
        result = F.median_filter(np.zeros((10,10)), np.zeros((10,10), bool), 3)
        self.assertTrue(np.all(result == 0))
        
    def test_00_02_all_but_one_masked(self):
        mask = np.zeros((10,10), bool)
        mask[5,5] = True
        result = F.median_filter(np.zeros((10,10)), mask, 3)
    
    def test_01_01_mask(self):
        '''The median filter, masking a single value'''
        img = np.zeros((10,10))
        img[5,5] = 1
        mask = np.ones((10,10),bool)
        mask[5,5] = False
        result = F.median_filter(img, mask, 3)
        self.assertTrue(np.all(result[mask] == 0))
        self.assertEqual(result[5,5], 1)
    
    def test_02_01_median(self):
        '''A median filter larger than the image = median of image'''
        np.random.seed(0)
        img = np.random.uniform(size=(9,9))
        result = F.median_filter(img, np.ones((9,9),bool), 20)
        self.assertEqual(result[0,0], np.median(img))
        self.assertTrue(np.all(result == np.median(img)))
    
    def test_02_02_median_bigger(self):
        '''Use an image of more than 255 values to test approximation'''
        np.random.seed(0)
        img = np.random.uniform(size=(20,20))
        result = F.median_filter(img, np.ones((20,20),bool),40)
        sorted = np.ravel(img)
        sorted.sort()
        min_acceptable = sorted[198]
        max_acceptable = sorted[202]
        self.assertTrue(np.all(result >= min_acceptable))
        self.assertTrue(np.all(result <= max_acceptable))
        
    def test_03_01_shape(self):
        '''Make sure the median filter is the expected octagonal shape'''
        
        radius = 5
        a_2 = int(radius / 2.414213)
        i,j = np.mgrid[-10:11,-10:11]
        octagon = np.ones((21,21), bool)
        #
        # constrain the octagon mask to be the points that are on
        # the correct side of the 8 edges
        #
        octagon[i < -radius] = False
        octagon[i > radius]  = False
        octagon[j < -radius] = False
        octagon[j > radius]  = False
        octagon[i+j < -radius-a_2] = False
        octagon[j-i >  radius+a_2] = False
        octagon[i+j >  radius+a_2] = False
        octagon[i-j >  radius+a_2] = False
        np.random.seed(0)
        img = np.random.uniform(size=(21,21))
        result = F.median_filter(img, np.ones((21,21),bool), radius)
        sorted = img[octagon]
        sorted.sort()
        min_acceptable = sorted[len(sorted)/2-1]
        max_acceptable = sorted[len(sorted)/2+1]
        self.assertTrue(result[10,10] >= min_acceptable)
        self.assertTrue(result[10,10] <= max_acceptable)
        
    def test_04_01_half_masked(self):
        '''Make sure that the median filter can handle large masked areas.'''
        img = np.ones((20, 20))
        mask = np.ones((20, 20),bool)
        mask[10:, :] = False
        img[~ mask] = 2
        img[1, 1] = 0 # to prevent short circuit for uniform data.
        result = F.median_filter(img, mask, 5)
        # in partial coverage areas, the result should be only from the masked pixels
        self.assertTrue(np.all(result[:14, :] == 1))
        # in zero coverage areas, the result should be the lowest valud in the valid area
        self.assertTrue(np.all(result[15:, :] == np.min(img[mask])))


class TestBilateralFilter(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test the bilateral filter of an array of all zeros'''
        result = F.bilateral_filter(np.zeros((10,10)),np.ones((10,10),bool),
                                    5.0, .1)
        self.assertTrue(np.all(result == 0))
    
    def test_00_01_all_masked(self):
        '''Test the bilateral filter of a completely masked array'''
        np.random.seed(0)
        image = np.random.uniform(size=(10,10))
        result = F.bilateral_filter(image, np.zeros((10,10),bool), 5.0, .1)
        self.assertTrue(np.all(result == image))
    
    def test_01_01_image(self):
        '''Test a piece of a picture of a microplate against the reference algorithm'''
        data = ('SkM5PDc5QkVBREhBR01lW01NP0tKSE1JNzY8PT5HQkdQSkZZU0'
                '5SUlRXUFJXU1NbZWxvcWhrbHRuYmBXV1JbT1VpW1pkaoOBjo+O'
                'mJyWoqfJyNPXyb58b36MkY+Fg4KIio+coE1IOjtAP0BGUVNJPU'
                'pYcGtHOkxUUUtDREc9NzxEPTpMR0FRVEtNRkJNU1JUVkxIV219'
                'fnttaGVpZVxYW1JPVFFQVFdiZ3WKjYyQhIyRoLe7ztDZ2q+EbH'
                'yVlIiGgYJ5dHeHmKJOSz5BRURDR1BUT0tKTFpbSUhWVlZRSUpO'
                'Rzw+UVJRT0ZCT1VLR0JGTlFPUlJWXWp0enp6bWRdX1tWVVhXUV'
                'hWUVFXaXJ4gHt1e4SOrq/IzcvNy8RzcnuMlpCHhHt8eHV0fIKF'
                'TUxCSD9DRUlMT05PTEhNUktRVlJSUU1PUUxCQlFZW01JTU5VVE'
                '1NU09LSktlam51dnZ1d2tgWFhXV1pecGhsaF9bYHN2dHNvbn2g'
                'qsbH0M/M0aeLY3yRlI2IhoSAgX99e3x4dktKQ0s8REpQT0tGRk'
                'RFS05HSU1NSU1LSkdDP0RITFJITlRESFZOUFJMU2Fmi35taWhp'
                'ZGFlXFZYXV9kaXdxdHNxcnF6cmxrc4GYyMvM0MzOwrJwaIKTl4'
                '2FhYeCfnx7fX6AfntKRUVNRUpTVlNJQT06PEJEP0JHSkBISUU8'
                'NTZCQEBHRk1OO0RXUExMU2yBgYl2YVxiZ2NfY15cX2Rmam5nZG'
                'hue4V+emllboGbr9jN1M7Lz5l4XoCRko+Ih4eEfn16en6Cg4F/'
                'R0RIUFBSVVVORD48QD05Oj1DRUhBRkVCOi8wPz09RkpMSERRUl'
                'NTXXOOlIl+dGloa21qaWdlZGdpaGltbWtxeYSId3BmaYGVs73T'
                'xNvPwKxpa4GYjYqGhoaEg4SCfn1/gH14dkZFS09TVFJRRz08QU'
                'ZBOjtCRkVFQj5AREQ2MT5CRklVU1BZW09aZXSLmJGJh4J8dmxl'
                'X19paGhqaGhqcHl5goqGeGlqdoKlrsfEyMTOzJZ4a4aak42Jh4'
                'aEgYKIgH19gYOCgYFDRktMU1RUUUY7PEY+QD0/RkhFRkE5OUdP'
                'QjQ+TlFOXV9fZ1pue3x9fnlyeH55cWxlYWBiYmNkZGNkaXFxc4'
                'SPgWdcbIuaxMHQxcHGt7xpX46UipCJiIiJhoOChIKAhIyWnaWs'
                'Tk9KTlNIRUhCODxMOz9DSUNFSj9LPkRJWFY2PVRTVFtdXmNvgX'
                't0cmxtdXl5dXF1a2x+cGxxZ2ZgaGFlYHF0gnFvcnqaqL3CvLrA'
                'xJ+Ab4WVj4WHgYOAiYGGgoWHh5eVpKuzsk1KRUZJQ0A/PTpASE'
                'JBRE1EP0pJTj9MXGlbP1BTVFRYYWpzepCQbWJocXd9ZGqJnJuR'
                'rbapo4N4bW9pZXJ7dHpzdYyZub3DxsfSwaxrcH2Sl46EhYKEfY'
                'F9g4GDlZimoaGlqbJJQkE/QkA/PD1ARUZGQERPRz1MV0hJVFVe'
                'XUxWVFZZXWh4gYWSoG5lbnV1eYSQubzEws3FpqN9dXBzeXd7f3'
                'Z4foCnstHKzNHL0KKCZHmPk42IiouNj4WAdoCGjKKlr62dmIuR'
                'R0JEQUZDQ0RFRkhGR0RDS0lEUl1FTlNLUlZLTVBXXmVteISMqK'
                'xqYWZ0iZK6vtfLycewk3J8Z21ob32EfXxxd46Sv8PMys/Wt6l7'
                'coaPkoyDiY6Pl5iMfnaBkpugqre9p52MlEdGTEpKRUVNT0lHSU'
                'dLRUdKTVdbSElMVVpQQEpOVmNtcHWFmbSfZWhwh6620MvNxK2g'
                'fnRjb2VtaWtze3R0doKmqs/NyMzKwpCFdYOPi4eFho2QkpyWiH'
                'l+jp6imqq9xK2jlqRHSU9QTEhJVFVNSk1KUkhDSlBWUkZIRUlS'
                'UElQXGJveHd4jaOni2V2kajMzOHSrZh+hHBrYWVja2tvZmxlcY'
                'ufwr/OyNHRt5pyfIiSh4B8gYWLkJicjYOAlKSooJCZrb27uaSg'
                'RUVHTExLT1JSTktJSFJFQEdLTklHTkJATVdaXnR5foKBgoyXk4'
                'Z4g7HA19vWwYRnY3t3aFxcaG5vdWBpboGxwc/GwMLJw5mGeYeX'
                'jIR7eX+HiY6UjoiNlqysp5mIjqGxx87BtUI/PUZITE1KSUpHP0'
                'NLQD1CREdITktDUF5aYXGDhoeIh4WCe3aJnqbZ1NDSq45bVGhx'
                'd25kaH5/dXhdbI6fzs/LwLrJs6WAiZWXoYKEfHqDjY6LiYCGna'
                'm3qJ+SlJyrprG2wMRHSktOTk5STkI9QEA8Tz84Sk9OWldgXFNe'
                'Znt9mYiWmYGReGhvibfG4dnAu4+AZ2tkam18bmyBe2RvZH2jtc'
                '3DxsLEwZx8d5annIqDiYaCiZ+Wf4qRnKWoo5uQiJCXlZaRocfL'
                'REVERUdJSUNBQEFDRk44MkVMTU5WXVxXZXOPkJuWko96inp/lJ'
                '/i6evbp66SjIWHdHBveYF9cnJjbX2bwLfGyMzPv651fZKko5WF'
                'eoiNjo6bmYJ+oKemmo6QlJqTkomLjJ63tUNHRUVASUQ/Q0ZBQk'
                'VMPz9ITk1GUVdXWnSEnJ6TmZGXjZqIlaiw6+2+pXuQkpGSk3Vx'
                'dX2Agmd1eXaPpsbAxMLJz52CcomhopSJhYKKipadnJKJk6usop'
                'OKj5aakZCKjY+XoptDSUhHVVxNP0dMRUdGTUxNTVFZU0pTTmCS'
                'mZmhlJB3hoGeoa3Gy9jUjnRygY2IhIFpcHqCgYJkdIqNoq+zws'
                'fFuq9ucpKUmJWHgoiMjIueoJKNma6wo5aUmJyYko2PkJaal41/'
                'RUhIRW1tVUBHT01RT09KS0lUZmVCUE5ws6yZpp+SdZGCoq6q7e'
                'nZxHpzfIR/eHJ2bHd7fHd8eIeqtLGwrrvJzZ6HaYyclY6KhImM'
                'iYuWppWIm62usJmKkZ2emJKKi5KaoJiBcEdMTEpbXFBFTFJTUk'
                '9LQ0lOWGReOk1agr+worCbiGCGhcT09fHy98RxeHiJenhweHd8'
                'cGxteZKjxc24srW6xbp3fY+alZOOhYWTmY+Tkp6am66xqZ+RjZ'
                'acmJSViJKfnZaNfH9ITUpJP0JFSE9SUUxHSURNU1VVTztScpSt'
                'pKy3nJFgkaLe+PT1/P/EdHl+jomKeXh3c2Zse4OgqLi9wMW2s5'
                'yNdZKlmpaTkIyMmKCelYiTpq6vpKSQlZmal5OUl5OdqZ6QhnqJ'
                'Q0VBP0A+QkVFSExKRU5LTk1JTU9CWoyjmJS1vJ2XWJO79/fv+/'
                '/mtHl5jImioYJ0b2tieoGMq7O2rb/KsJ1ibZijn6OckI+ZmJWa'
                'oY6NmqWqqJiSlJ6fl5CUm5upop6TlI13fjY9PDc9PEFEP0FHSU'
                '9WTVFHS0xQRV2psJGUr7OQpmSX0PXz9/z/vqJpeJuuppZ2dm9n'
                'go+Rk6egrrGxu5uBdISVm5eVlJWTnr2vmpSanKOrrJuPj6GikJ'
                'OFlaGXpqGVjYuCeXdEQkBCRkVISEJHT1RPVEVKSE5NVVV0uLJ2'
                'h6iod5Fwp+r18vP20bioX2isvJ+IY2dyepOYlZekp7SznZZ2eo'
                'eVmpqZlJWMkJ60rqGPpa2up56SkJSgnouJiJipppmYk4p/f4iS'
                'TkVCSkVDSU5PU1JNT1JDSVBUSldqh7OkZo6unnKHeqjy7fLwtp'
                'WyqGdorrqTgF9keIORjpyiqrCvqIl8a4SgppmRko+QiZGVm5+o'
                'oa2wqZmSkZSbk5KJi5ukrq6Uj42GeIKVok1HQkU8PklRUFNUUU'
                'hQSE5VVUNTdo6jj3SdtqJvgISv+PX3+XeNuZxucY2TfXhnb3yJ'
                'lZSYrLCvl4x5eoeXp6KPiIuLiY2Wi4qOqLSyqJWLkZeYmYqKip'
                'Ozs6+qnYyHg36Jm5lKSkI3Nj9KS0JFUlw/TU5QUlI/VHqUlH6L'
                'obGqYoCZzv7+89Vjmb2San6DhoB/eHh/iJOXmKummnhydoiimZ'
                'SOhoeNkIqTk4mXmaOyqZqLjpubkY+JjJWiwb2rn5eHgoeHkpmO'
                'Rkw+KzM2PUI/QkhMQU9VT0tSSGCBkH1zoa6uql2Iu/D//+GfZ5'
                'm2mWqHi4qIhYaIiYmIiKOkinhqcYecp5KFhIaLkpeTkYWGrLCq'
                'rI6LjJefl4yOjJmwvce8no2Df4WOjpWVjkBFPC0xNDxEQ0NEQU'
                'ZQW09FU1BnjY5rc6e3rKRZjtb////xmHCbr5lqeXt6f4CLlJiU'
                'jY2ZkHNrfImfo5eLhoiHjJKXmY+NlbGvqaJ+ho+Vlo+Nk5uwzM'
                '65rZKKf4WNk5SUko49Pz04MT9PT0E/TVpETV1PP1JPYpOUcoGe'
                'saikV4/l//7/9I1imKGGZ2xsdIqMlZuZlJCVe3Ztep6lqZmAhI'
                'qKg4aRmJOSp7CvnZqQhpCVj4uKjpWzx9vMnZmQmIiQkpWZlI6L'
                'REVEP0VLT0c9SFRTSlxZWVdTTW6Yh2Fyna6xqFeT4v75/+uHZo'
                'l5bm1icHqQj5uXmJyTlW1pfY+zs5mRi4uHiXt7kYuYqLmynJST'
                'hY+TkZCDk56oz9i9q5SdnomBmZGSiYeJmT9DR0ZJSEdBPUxYWE'
                'RSUlJMTlBzoJNmbaCzsqhTl/D78Pv/k1uCc2dpa3yCkpWhnJiY'
                'i412epScrKCKgZSVgH+EfoqcqqurnIeHjYmJj4SIiJa60NzBna'
                'SblZKQkJWCh355fY9ARElORkA+PT5NVlE/SFJWRUpVdainc3Wr'
                'u6yeYJb6+PT2/ZxSgHloaXWKkJKVm5qWlIaDeIilqKCUiIWJlY'
                'yKjIiUqbmrmoh5fomNjZmFiJyly9bFpIqgmoyHlJWJd4KCgo6d'
                'QUJHTEY/QEJBR0lCOEBUW0VIVnOts3t6t8Wum2CQ9vfy9/+pXH'
                '97cnmIlJWYl5mZnZ2VjpWitauOhIaIj5adkICTsrm0pJOJhYiK'
                'i4uVjI63xdHJrKCQlYmLiY+OiHp9fIqouD4/RUpKQ0VHPz0/Oz'
                'Y9UlpBRVdysLB3dbHEsaFVnPn88fn9lGx7d3mKlJSNmJyamJeY'
                'm5uytbCdfXuKkpOUnJiVrby4opuUlZmXkY2Mi5yjyc25s7Owno'
                't7jYyGe4aDfIGgz906P0pRQztBRTw8RUk+QlBVPkZbe66jdXqn'
                'taSQRojv+///oWxvfYOJiouMjpOenZqQj56mu7SbjHl+jpaNjY'
                '+kxMqwrZWXlpqgnZWSjo2zxNG8lpyorKiWe4N9fneAfoOhwNTK'
                'PURMVUA5P0ZAQE5XT09VVUBKX4GtlXWGo6eYfD5fuszt2mdseo'
                'mXlYmGiZWQm5mcmp6wtbesioSAg4iMk5GXp7+9paiQmJWTlZOO'
                'k5Kewc/CrJGZkZelpIZ7b3t8f4GSt7mxnz9GS09NQ0pRSUVRV1'
                'hXWFdETV6BsI1yiJ+hm39YYYefiXOCeJCVm5iRk5CRgoiCkaCr'
                'ubadlnaCi5GPkpGYuryppJqTkZmVjIqGh5Czx8vCopyWlJKLl6'
                'CNgWtxdoSSp7+qq7ZFS0xZUlBJV1lWZF5VT1pYQkRpi6uXgJCt'
                'r3xsXnB8eGhhanuNn7CmoqyDfW94foGstqmjh4CCiYyaoZuOp7'
                'mzn5GJiJqcj42GjY2iydC/rJSQi5KPjJSMh4Z2dX6Isqqnq7HC'
                'TFRXXltXU1tXU1VQRE9aVEZMa4awmnuTsKyDdGp0eXVqaXiKpK'
                'OkpqetfHJ0fZ6lxcekm4eGiI6IjaCisbezoYyIj5qcmo2WjIyp'
                'wNC/o6CblH10goGEhYCAgImTqMbAu7rFyU9XWlpaV15eVVBGQ0'
                'hJTFFKR2KPtp15mbGnjX94eHVxbXeJmamqqrGgmW9xgY7Hzc/A'
                'lIyLj4yXk5Kmp7+1o5SLjpadlZSMmpyfy8+9qJWblol0b3l3cn'
                'd0coCPnrXDwbu1uqpKT1BQSE1gW05NQEVKR0NMTEZejrqkfpqr'
                'oJOFf3pycHeImaOgrq6xin5peJai1NW7oYeIkJiNkpedtbS4qJ'
                'WTmZ6alo+SkJW0xtfFlpSUnIh8d4B8emxwcHGFjaKvvsC5r5+I'
                'R0VIRzlEXVpKSkVPQk1MSU5WZ4C4qYSYn5eSg3l0c3mJnKitpq'
                'qalHFzfImxtL6/nomNj4+Uh4WTp7u6rqKXmZ+hmpSSlaChy9rD'
                'rX2Jlp2De3uAfn9wc3qElZGdpsXItqWDeU5HREM6Q1hZTEpOVE'
                'hISktNVW6LtqWFkpSOiH5tcXiFl6mvrqeigHRldp2rysCqqZSM'
                'lpGNiYGIqL2yopacoaGem5eUjJW3wdXNnpB3i6OlhHt7e3p9dn'
                'iDkpmSnqnNyaiTcXhWTkpIRklOWFNNUU9SQkJOSElsm7OWgpCN'
                'iXt5dH2LmKWwr6iZjXNvdYm9ycm+pqSam5ePhYGNnrzDnIuFlJ'
                '2dlpSVlJWgx9a/spmLeI2nqYV9gYJ4e4CDiJWWmKqzwraTin6H'
                'VlNTU1JOSFdXT1JHSUBKUEJIb5exjHuMjIVyd4aSnqeusqugjX'
                'tpdpOi0M61sKmpoamWknuCpK62rYiTjpaWjo2RlpWttc3PnKKw'
                'nH2Gl6CFhYiHeHuNkZObmaSusKGVf4eUmUdITVZYUkJMYFRHWF'
                'JPS1dZWHCLr4uHhH2Cf36Kkqynsq+NlV5lcYi0w8/FrKWtpqKn'
                'k36JnbiroJ+RjpKThpWFgZWsytXErZaboa+RjpaUg4OIgXCDjo'
                '+Yk6ijsquEgoCPoKBLS05TT1FWXVNFTFxRT1ZfVFd4iaOHgYJ7'
                'foOEoaSspamre2xebIeg0ti+qaGqqqOtpIiJrLe2ppeUm5mak4'
                'KKjpuut9HCpaGdoJaXk5GUjoCHi4B9iZGQnKS2sYuHb3yPorS3'
                'TEtMTUxGVF1KSVdXTU9eYU5WfIWYiYCBdneMlaekpaWamGtYbY'
                'CyyObVnJKjrq+praCRocG/o5iTk6afkpGHiZuxyczCrpCSlp2O'
                'i5icnYx2gJCNgoubmqizs6t7cHyPlaLH2EtLS05LQUhRSlJfVk'
                'xUXVlTYYWNlI19fXN2maqvq6euiXZfZYaX2+TUtYSXs6qrpZmf'
                'q7a9rpKRm6ClmomNlJiywc/KqLKmnI+WlJaVmZmJdXyKjpCctb'
                'W4uaGSdHWQl6C43t9LTU1PRkxQTUtNU1dQXVtUanaQm5GIdnuA'
                'iKSytramqHttco2ksezgr5yRr72rl5GSprm4qpuXlqCnmpOSka'
                'Oxzs2/q5OssaaOioqSkZCQjYSFjJC6wdDGtqyNgn2Tn5XI6s2n'
                'TkpHSVBcWVBNRkNKVGBcXIWLl5+QhHaEm6esr7a3lIlwdY+ny9'
                'blyZqfrby+tY+LqLaxr5KQnpudopSYmJ+5yNPFqpuIlJukl4+B'
                'iIuLioqIjJem29fNt5yQgoKRq7O6+/WehEhCREdcXVVNSURAQ1'
                'RbaHWblZ2cjIWHlqy3qqq0r352eICerejuyLSjp7O3uLecorm0'
                'p6KIkZybmJqZm6Szys6/sJaSjIyKm5+fjoyAgoCJnbTE0c7GrZ'
                '6EfoWPlqTY8PDLjppAPkVOW1VPSD09Rk1UWHSNqpyil4WImKGt'
                'saOpmZBodpOausvt7aiosaaosa+trL2zo6GWkZ6YnJmXn5a6w8'
                'nDqqOEgIuOg4ePoJOMiIF+n9r///2yrpSTgoCRnKCl/v+xmZGd'
                'R0VETFVRU0tCUFlhaX2Wp7Con56Pk6+pn5+vnYt7a4mamMfX68'
                'qpvci5nKaqqLq/pJqQnJWdl4yMiJ6axsW2tJ+Sg3uFjY6MjY+N'
                'k42Ni6L///fnrquIioybop7U6frMf5OQkkhOTlBVS09OS09aZ3'
                'yUrLCqqKKaoJ6qqqadmJNueoWfp6/o8My0wNnbw5SVqa+zr5OR'
                'kaKWnJaNkpe0u8PCrKOMhHx3cn+FhoWFhYyMlK2+59DAt6effY'
                'icnqy+//+ylYeLkJVET1NXVVFUT1FcbHygoaejmZ+hm6Ktramq'
                'rYl7YHuSnq7F//6tmsHKwLKZo7C3pJ6QlZymk5KRj52nwcq7ua'
                'CSfHlybHF8goSGhIGEgJHAysGxqquUj4Sbpp3L9P7rgX6UkI+Q'
                'RktUX19maFNddIaRoJejrZ6XmZemsrOwoKJxX3yPlqHO5v//kp'
                'HJybOtrbGxsI6Sl56enpSPkpqwtb6+sKiNiHt5cWt+goKEiId/'
                'foqezcyurZqeio6Yoqq8+P/ApoSOkpqOjFdWWGZqbnVngZSYko'
                '2Ur7ujl6Cqo6CjsIh9anWYoJip+P///4mZx8e1s7ytn5l9jJuc'
                'lZCVlqexw8C+uaybfH54e3h4f4N/foGAf4ObrcnMsaiFj5Kenq'
                'HR9/rNin+Pk4aYko9pZGBlZmV3gq6wnI+EmK6xoqWusZyelJZr'
                'bYGapqWYqv////+KmrK2s7m5o46OhJWYmJCTlqG9wsnCu7qyoX'
                '56cXd7f3mAgX9/gImYqK63wqybgJqepqzJ//+0lH6MioqOlJWP'
                'bmtmZ2VshpfCt56Sg5enq6+0raOnr4h5an2Xo7Gpp7L/////h5'
                'ast77BpJqNj5OakZmXm6Wy0M3NwrWtq6yShnF2fYB9gYOFiY6Z'
                'p7q2ra2WkI6lpq7X9/3NfoaGmIaLmZOVk2xsa3BrgqKfuq2ck5'
                'SfpKasq6yxpKNmZ4WXqa+uprG69/z85ImTqrCxsoiXk5Kal4id'
                'mp7Axd3W1suvnpWspJV8f4KChISBh5KaoKi3tKWbhZWkp7C+//'
                'y4ooOOjIuJj5WQmZ1sgnmHmZy1rLKem5eZpKypq6untJyDb3CV'
                'oqissbDBv9PX48SHnLC0paCIj4+bj5eUjaOxztDT4OLAoaCNmK'
                'aklIuKiHx1h4+Wlaiupqd+eISXn6TM2/7Vh4iCi4qRlJSSi4aJ'
                'gYFxkbTJ/valqJSboaCnpquxqKB7cXuFlp6qp6+vu7nRzM61mK'
                '2zq5iXjJScnIaPnaC4v83U1tvVu6Gbh4uRk4yJh4F7fpCYpai1'
                't5eTcn+Zmqe8//+/oIyHjZSSlJKQkIyKj4ePm7z7+Pj/05aana'
                'Keo5yao5+VaG6OmZWYqKGlqbzA1MrFuaW0pZuOlZaalZmNlqeu'
                'x9HQ0c3JxLmnl4eFiIyNkI6FgI6cnKOipqmJkI2hqKTO5f/pjY'
                'GVkJOWk5CKiIuKipCRvvP7+Pn8/senjaCbmZyZoKGSj3eBmpuW'
                'mKahjZe+yMS4u7mrr5WWlJudm42aoqq6ucfM0cK3sLK3rp6FgY'
                'GDhYqNhpegoqKpopeYdoqlraa7///Cn42MkpeSkY6LhoaIiImP'
                't+H89vfs+//MsJyelpiXor6wiYeSmp2VnaGlpYyZz9SyprGzqq'
                'qTn6CempqXoq21ycTFvca1pKGhpqSeiYSCfXiAi46lo5ymsKWN'
                'h4uWoqXD5//ukXuZl4qSjYyGh4eKi4mJk+//+/n08vn88a6eoZ'
                'ienae3oX+InKKelKWoqKyir+HarqiztJqfmqiimpSfmqa2tsnE'
                'ysW3raGdmZSRlJKNiH12fpCYn6Okq52Nf4Sio5yz/P/Np4GJlY'
                '+Qi4yFgYWIjY2Lj5z///H38/j49vz/iIGbnaailoeAj5aZoZyi'
                'o6ussLvSvqespKSOlpmgmJmSo52rxb7NyM/Qrq2hnZ2UiIiJh4'
                'R+dX2LkqCprKaEfoeXlp242f/vl4WElpGKlYqNhYGHiYyMjJiv'
                '///8+/n28/j595uMmpWko4qKkJOPj6emmZirqbW/vp+hrZOSk5'
                'eZmJGbk6Guts3H39jSx6+vnZikn4qDgYGDhICFiYmdnZWOdoea'
                'p6W1+//VtX2JkZWcj5GLkIuHi4uMjJCkwfz5+PL3+Pv//vzKjo'
                'meraSbmJSGj5+al5uhu627uaWhm5qUkJick5aVlquvsLnIzeXm'
                'wLilqqWaoKJ8jYqMjoCFgYWAipJ3fHudpaG10v/6npOJjJGXmJ'
                'SPj5CNkYOQl5iiwsX++/n5+fz++f//w4yKl6SjopeIh5SflpSg'
                'qcW/uLGdmo2MjZWSl5CPkpqmsLi21uTgzK2loqmjnaake4SAjJ'
                'SMj5KWjX6MdIqanZuu9//ntX2Nl4+UlZaVk5GQkIeNlZWsusXA'
                '7/z6+/X6/vr//7Waj5alq6iUd4KboZKRpqq8vK6onpqKj4+Ri4'
                '2KiJmloqvAxff/272kmaGnoaCopnyAhZWknaCjoZR/jYKZpp6z'
                'zf//rJSJjJSNl5SUlJSPjYyLmpiYwsrEw8/z//n++v////ayoZ'
                'GXsrabiHGBmp2QkquprrClop6YjpyVjo2JjJGpsJ6lu9H898Gz'
                'o5uhp6Ggop6BiZKgq6mppqCSfY6dn5qt8f/iyH6Kl46Vm5WVlJ'
                'KRkIyIk6CmrtDHvMS12/////n+/t/GppWZn7izfXZ4h5SVkJax'
                'rq6xqqOgnJCYjpKSj56qtbKfqrnM3MmkqZueoqmhnpaSg5Seoq'
                'SqpqCXkI6aq6Kqzvz8qaGGkZaZmpqPlZWPj5OQipqov8fQwb/C'
                'qb7o/////PTNwrCSrLKxnmZqfYqQj5KYrq6vsq2nrrSmmH+Tl5'
                'ywubCmqLXO0tLJsq6RmKaroZuOiYmbqqagpp6akpWhnZ+q5vzg'
                'w4eNl52XmZKOkpeWjYmOkJCtu8/Iwb/Iw7C4yN///fn4+//Qo7'
                'nBmYVibX+Ij46ZnKemqKivs7/HxbGEkJyirq+bmrG209fh5sS0'
                'jZSwsaCekIyLmaupn5+SlJOZmpqyy///uJuOjZCWk46MlJKTkY'
                '+KjZimxcfKvb2+xb67xLG0t8rr9vz/wrO1wHxvbHd/iI6On6Gl'
                'o6Ofsr/Dw9HPl5KipJ+cipa3r7vL5ei2p42Vt7Wio5mUjpedpJ'
                'uViJCVoJ2z9fzpzIyDjZaLjpqSioiPiYyTlJWsxs3GwL7MwLOu'
                'tMbEt63T+fz6/ui9q5p9cm9xfH+MiaawrKSLjKm4v8bi57+iqq'
                'yTnJ+mztDMsMDPnYaLn72ynaCbnZeSmqSKiIaVmZK10/v/spWX'
                'koqOlY6OjouTmJKGi4+ly8zHwr/NycSvqKu7tLrJ7fv7/P/y2Z'
                'iKcm5wd4SIj5WspZaSgYSYq7zN6e7LtZ2fjpmgtu/69caRqpCE'
                'kqO9rJmbmZ6SkZ2Pg4OTnpy59/7yz4aCkZmSlZqTk5ONkpSMgo'
                '6rvsnGxMfFy7uvmZOyvbLY9/j5+f3//9iQhXV1fH+IiJGgs5+M'
                'jIKFiJKht9TZvbGNmJSjpsH8/v/PaY6Gg6W3taGZm5igjpSRiI'
                'yVnKG84f/4t5J+kZacmJiZk5aYj5GTkpChx9PDwsLKx8KqoJGP'
                's7as4Pj4+Pr///TZjId8gIaDhYCYo6iWjJCLj4iEipqytZ+YjZ'
                'yfqqvG9/vlum2Sh4Kyv6eaoKKanYaSgI2SnqCw/P/gv4SFk5+S'
                'j5yYl5GVmZKUlpyuuszPv8XFyb2vmpyYm6zAyO/z+Pj9///iz4'
                'SCeH2BfoF+oKCakZWWjY6Mi5GYn56RkZqgnZ6mzPv8zZmBmIyJ'
                'srCcm6utnJZ/ioqalqDE3f/1qpSBkpeVlJaYlpaPlZiQkaGmxM'
                'vBw8DGxMGtnZCbnaCn1/L5+fn///Lu4b5+fHN0dXV8gKChmZyp'
                'oIyFhZKjnpGOj5adoZOPp9b4+cR9hJOQmaqcmqGtrJiQgYmSnq'
                'u+///HsIiHipaOj52cjI+Tj5GTi5CztMrOv8PIw7SwoJiSn5+f'
                'r935+P7+/ubL2t6/dnhub29xfYSbo5ufraSTjpGfqpV8gIqUnK'
                'KMhKjc9vS5cH2RlaWfnqKopKGTlJGXlqfW7//qmo6OlJGSjo6R'
                'iYaKj4mLkpKewcbFxcTFxLeen5eZlZ6dnMvp/P///76S2PzQuG'
                'luaW5wdIGHmZ+Ripibn6ausayIa3iKkp6kh3qn3/f3rXF9mJai'
                'kqeqrJyZkZqkpcDS///BmoeLkZmPj5OOjo2Ii46Gh5WgtcTSwL'
                'vEv7Wmk5eVmpOXk5Ta7/P//4eXwOv007dpYGVvZmp/i5uPc4mf'
                'pJ2puayZjHl5i4qdmoiBjcb388WHi5eilpOhsKOOmpCepMT6//'
                'vaiYuLkJKXl5OQkJCRj5aZjZemvMPCwr/Eu7qnlZWVl5ycmp+l'
                '7fX+/qSd7///+diyZGFobGRoeYahnIKLi4eMpLaokod/fIWEiJ'
                'iTgYG3+PvCl5uhop2WnqOphY6Undr1//GxloGXjo6Ok5WTkI+N'
                'jJuUkZGpucfEwr6/ybeomJOXm5ydm5+lqfz8//uawf/////kqV'
                '9laWhlaXWDpaKHiYR4eZS5rY6EjYiFhYWWlo6VyPLvs6Srqp6k'
                'oq2qnnqIvtL//NK4hn6Ono+TkJOTkY+Pjo+YkZWkwMfJwsC/vc'
                'KsnZSXlp2empigpqb///fGpdb///z83aZcaWtlZGp3iKSfhoV8'
                'cHiYyL2RhZqZjpCJj46Vq8jKtKarsqySn6i8sJ2vwf3/5caIj5'
                'CTmJSPl5OTkY+Njo6SjJOrv83Ew8TCw7iumZubnJebm5qanp2b'
                '18+boaLY/////NmkWmlrZGVteougnoJ6aWF8rdvNloKhpZmekp'
                'WWnKapmpOlr7GrlqKnt73K///95KCXi46ZloyMlZKPj5CPjIqK'
                'jpemwM3NvMDJxb2roI+YmpqbmZmeoZ2Wk32TlqOl1fn3///bpF'
                '1na2dmbn+MmZR3cmJcdabj2Jh6mKWeoqCcmpealY2eqq+pp6i0'
                's7Do9//mtpqAj5aQk5GJlJuNioyQkY6MjZC2vMnIxr3Ey7qomJ'
                'yTlpabnZiXn6Cbko94mIyUmM78/Pr+0KViZ25vaXCBiZONc3Rp'
                'X3Kc4dubdI6gnaCjmZCLmpiMnqWpoZywwcGw4d65nYeHjJGMj4'
                '+UkZmVjouNj5KRlJ2oysnIwMbDwbqll4yWkZWUmZiZl5mZmJOR'
                'bIltj5zE+fv7/s2naGhyd21ygoeSlX58al55rd/bnXKHnJ2dpJ'
                '6WkqajjJaao5yOqLzEr5mflIePnpmOjpiPkpKSho6QkI+Rk5+y'
                'xMrKycDHxbunmJeRkoqVlJKUmpmUkpiZl2Z7ZXeSyvf5+v7Ck2'
                'R0c3N2eYePmYyKf29mgLXk3aJ2hJmdqb+pmJqgm5GQjpKShZOk'
                'r6Z+g5SelZGUkJOTkZKPjY6RjYqMi5mnwM3NyMLOyMCspp2ckY'
                '+YmZKUj5CMh4qRkotid2R+lsz5+/v/xJZqc2prcXiGi6aZl4+D'
                'eYet7OOicH2XpbayoJecmpKQmY2Qj4SRnaKVk4yOkYuLj4ySkY'
                '+PjIqLjo2IipGpt8fIxsbCx7qwnpqXl5SWnZuTj5OSkJCRlJaW'
                'YnNjhpjO+vz8/8eZcXpwbXB4iZKupqWdlI6atvTromp3kKW2pp'
                'GIlJiQi5OVko2Gmaaom52TjIuHiY6Pj46NjoyKioyLi5iku8PI'
                'xMPGwLyooZOTmpianJiWlJGTkpKXl5eaoGBuXYKczvf5+v/Gl3'
                'KCf3lydouatbW1p5uap8X07qJrdYiWn5KEfIeOiIGBnJmYlKi4'
                'xMKnn5qYlJOSko+Pj5GRjoyMh5WtuMS+w8HIxretnJ2WmZSWoK'
                'SalpeSk5SVm56ipKdbalJ0oM/09Pb+w5ZwhYR/d3eKl6yxtqqh'
                'oKrB7+2eaXKBh4N3en6Afnt5e4+Xop6osMrYv7Wopqimm5aRkZ'
                'KTko6Li5emvsHFvMHDw7unnpWbmJiIk6ixpJqWkZSZm56lrq6o'
                'WmtScajU9PP1/MKScoKAf35+ipGjqrm6v8LBzvHum2NseX92bX'
                'J1dXZ7fn10iaKfnZetwMq8qKewsKOdl5WUko6Mj5O2uMTCycTH'
                'xrOpmJiUmpKPmKCrrKCWk5KWnJ2eo6ypn1t1XX6y2/f2+f/FlX'
                'F/f35+f42Up7HAxszS0+D19ZtgZnSAdXRtamtzfYOEcXyVnqGV'
                'm56sraaosKuipKOdlo+NkJyoysPFwszHxLmhmY+XlJiRkbOxqZ'
                '6ZlJSZmp6en6ChmpNce2eKueL7+vz/yZhsfX98eHqNmqavuLOu'
                'rrXL+fqfYmVyg3h0b2trbHKFlIR9jKC2raCQjKKvsrChmqSqo5'
                'eNjJarvcnCwsDJv7Wml5GOmZaZlpy6tKOXmZmYnp+eoaSfl5GP'
                'Y35ujcHy/////8GQbn6IgHBuf5Onuaqch4eZwfr8pmlueYaLfn'
                't2c3V9hY1+gI2ltrepn4WNm6qws66nrLCekpGpx8bKxcrJw7if'
                'pJmcnaCalZyotKGQj5WYmpucqq6poZiPj2yEcJDG9f///f+9jH'
                '2AgHx0cH2QnqmZjn+CmMD7+6hsc32DgXV1dHV3fIGGiImOnaWl'
                'paedioSQm6evtK6zrre8xs7CzMzOxrmrmJ6hnZygoJ6gp6eckp'
                'OXmJmcoKqropiSkZd7jnaRxPH8/v7/wJCBgYGDfXV9j5KWjYqC'
                'hpvF+/ijaXN7e3Nub3J0d3l9gIuKjJKSj5mms6eVg4OYqKuns7'
                '3MysnNysXJybqnm46Wm5SRmJ+gnZ2dmpmbl5WXnaOnoZiVk5OX'
                'eI94lsLt+Pv+/8OUf4aPj4R6fYmJi42SjYuiyP/4n2ZxeHdvb3'
                'BwcXR4e4GGhIaLiYWOnaaysZyGj52hp7G/xLy6wMC3vbemmI+O'
                'l5WTlZugoJyYlpicnpqZm6Khn5iVmpqWk2KAcpTO8/f7/v+/kX'
                '2Kko6DfHx/g4WQlo+Hn8P89p9pdn18dHJyc3N0eXt/gX1+iIqH'
                'ipKhpbW9q5SPlLCzuru2t7Oqs7SompGPkpiOlqCjoZyWk5OVmZ'
                'ybnaGnmpmTlJqalZVgeWaK2Pj4+/3/vI57goWBfoB/fYSGk5OJ'
                'gJ7B9PCfcH+Eg3pzdXZ4d3l4eoB7fISIiYeJnZqpvr+zo5arrL'
                'm2sLSxra+smpORj5KVkJ2ppp2YlJKZm5ucnJycnpOXl5SVkpWd'
                'coBljuD7+fv9/7uOd3p6d3d5fYGFhpOSioOlxPTvnXGDioZ8dn'
                'd7fHp6eHeBgYOFh4WEgYiPmKSxwb6vnqG2ubCvrbSkoI2PlY+S'
                'k5ikraednZybnqKjo56alZOPlJSWlZKUnXeDapzo/vr7/f64jX'
                'V4fXpya3J/gIKTk46JrMf+9J5xho2Lgn1+fX17enp7gomOjIiG'
                'gn6KjZGTkpuqtJ2Zr767s6CfnJeIj5mSlZWUn6Gdl5qdnJyjp6'
                'iinZeTkZGPlpyak5RwiWyT7f/+/fz9roF2enV2bWd0gH9/iJSO'
                'jq/Y/PmaaYCQgHuJhoF/gIJ+e3p6gImJgX+EiIuPjYqKkJSSm7'
                'O1ubCjm5CLjZKRkZaYk5OSlJKWnKOjqKOfnJqYnpOLjZaVkpOT'
                'Y3hllfD////8+62Bb3t4dGtpc3d7iJCNhY+uy/v6mWV7j4F5hY'
                'J+eXh9fX1+f4SMi4N/g4OEiIqLioqJjpausbeyqaOUjo+TkpKV'
                'lpSTk5WTlpqgnKCenJqXk5WZk5KSjI2XnmBzZZvu/////Putgm'
                't8eHBqcHx5f4uVj4OMq8j6/JhheoyAd35+fHZ1d3t9goSJkIyE'
                'gIODg4SKkI+Ig4aMpquysK6nnJORk5KPj42TkpOWlpaYmpmcmp'
                'ucmZCRm5aTkYyTnqJnemqb5v7///79s4hzfnlwa3KCgYqLk5mM'
                'hqbQ/P6SYHuHfHh6enp2d3l4eoOGio+LhICBhIOEipGSi4aDhp'
                '+nrq+wqpuQjpGQjYyIkZGSlZmYmJiXl5OVnJuTkpqUjpKYoqGY'
                'a31tntr2/f/+/rSLgIR7d3Jzf4KPiZGbkYWm1fz8imF+gHV4fH'
                'l1dHV5eHeBhIiNioSAgIOCgoaMjo2Ng4OfqrCytbCVi4qPkI+O'
                'i5GRk5ibm5ubmZaPj5ealJaYk5GZnqWdj216bqLZ8/v8+/yyiX'
                '+Aen57dX6EiYuSlYuLrNL8+4VigHpwd395cm5xeXp6fYCDh4aC'
                'f36CgYOEh4uQkoOAnamvsbewmY+MkZGSj4ySkZOYm5ybnJyZkZ'
                'CXmpibm5manpmZlpFzfG6j3fT4+Pf4r4Z5end+fnuDiYeOkY+I'
                'kK/N+/6GZIV+dHp/fHNvcnl+f3p7f4KDgX59gYODhYeKj5KFfp'
                'ypqa20rp6WkpeTk5COlpSUmJqamJqdnZiXm5uYnJ2dn6CVkJKW'
                'eIFvnuH29/f197CIe314ent7hYuMjpGPiI+vz/r+imWJhHp6fH'
                'x5d3Z7gIF5eXx/gX99fX+BgoaHio2NjYKeqKirsqyblJKXlpeX'
                'lpeWlZeXlpeZmZ2bmZuYlJWfm52hlpCUmXV5c6Lt+/78+/yshn'
                'V3gH1ygJCOj5iXl5GUu+T//olbf459en6TeXB+dnF/dYCHhoOI'
                'iYmJh4WHhoWGiH+CnqyyrbOulouKkZCPkZSWmZeUkZWanpqYmp'
                'uXl5qXmaChn5+clZRphHin7vv+/Pz9rYd3fYF8c3yIh4uTkZKN'
                'jbHY//uFXoaTgnuDkHpxenNzfX2ChoSEiYqGiIaFiYqJh4Z+gJ'
                'yrsKyzrpmKho2PkZWYmZyal5aYmpqVlJqbl5ibmZOZmZSWlpid'
                'a413qO/6/vz8+6qEeoKBe3Z6f3+JkI6OioiqzP/2fVmEkoJ7gY'
                'h3b3V0dnyGhoWDhYmJhIaDgYaMi4eEe36Ypq6psa2ai4eOkJOW'
                'lZaYmZiXmJeVlZWYmJOSlpWVl5SUnKivuGuHd63y/fz6+vmog3'
                'qEg3t5ent+h4+LjImHqMr+9HhTe4d+enZ4c3BxeH6Ei4mFg4aJ'
                'hoB+fH2DiIiFhH1/mKiuqrWxmI6Nk5OSko+NkZSUkpOTko+Qlp'
                'WNjJOVk5KRlaezt7g=')
        fdata =('R0dGRkZGR0dGR0dHR0hMSklJR0lJSUpKR0dISUlLS0xQT05WVF'
                'NWV1lbWVteXV5iZmlra2lqam1saGhmZmVnZGZtaWltcX5+h4iI'
                'j5OOmJy9vMPEu7CFgoaKi4uHh4eIiYqPkUdHRkZGRkZHSEhHRk'
                'hKT05IRklKSklISUpIR0hKSUlOTUxSVFJTUlFWWlpcXltaYWpx'
                'cXBraWlqaWdmZ2VkZmVlZ2hsb3aDhoaJg4iLlq2xwMHFxaGHgY'
                'WNjYmIh4eFhISIjpNIR0ZHR0dHR0hJSEhISEpLSEhLS0tLSkpL'
                'SkhJTk9PT01NUlVSUVBTV1lZXF1fY2ltcHBwa2hnZ2ZlZWZmZW'
                'dnZmZocHR4fnx6foOJpKW9v76/vbeCgoWKjoyJiIaGhYSEhoeI'
                'R0dHR0ZHR0dISEhISEhJSUlKS0pLS0pLTEtKSk5SVE9OUFJVVl'
                'RVWVhXWFlmaWtubm9ub2toZWZmZmdpb2xubWtqbHR3d3d2d3+W'
                'oLy8wcC+wZqKfoaNjouJiYiHh4eGhoaFhEdHR0dGR0dISEhHR0'
                'dHSElISUlKSUpKSkpKSUpMTVBOUFNOUFdVV1lXW2Nme3NraWlq'
                'aGhpZ2VmaGlqbHJwcnJyc3N4dXN0eYGQvb+/wb6/taWBf4iOj4'
                'uJiYmIh4aGh4eHh4ZHR0dIR0dISUhIR0ZGR0dIR0hJSUhJSklI'
                'R0hKSktNTVBRTE9YVlVWW2l1dXpvZmVnamhnaWhnaWprbW5sbG'
                '5wd316eXJxdoGSpsS/w7+9v5GEfIeNjoyKioqJh4eGhoeIiYiH'
                'R0dHSEhISUlIR0dGR0dGR0dISElISUlJSEZHSkpKTU9QT09VVl'
                'hZX21/hHt0b2pqa2xra2tqamtsbG1vb29xdn2Ad3Vxc4COqrTC'
                'ucW/tKB+gIiRjIuKioqJiYmJiIeIiIeGhkdHR0hISEhIR0ZGR0'
                'hHR0dISEhJSEhJSkpIR0pLTE5TU1NYWlVcZG5+iIJ8end0cWxq'
                'aGhsbGxtbG1ucHR1eoB/d3FyeYGcpbu5u7i/vZCEgIqSj4yLio'
                'qKiYmLiYiIiYmJiYlHR0dISElJSEdGR0hHR0dHSElJSUhHSEpN'
                'SkhKT1BQWFpbYltpc3R1dnJvcnVzb21qaWlqamtra2tsbnFyc3'
                'yFfHBtdIaSubfAuba5q69+e42QjI6Li4uMi4qKioqJio2Qk5eb'
                'SEhHSEhHR0hHRkdIR0dISUhJSkhKSEpLUFBISlFRU1daW2BqeH'
                'RvbmtscXNzcW9xbW52b25wbW1rbmxtbHN1fXR0dnuSn7O3sa+0'
                't5aHgYqQjoqLiYqJjIqLiouLi5GQl5qenkhHR0dIR0dHR0ZHSE'
                'dISElISEpKS0lLUVlSSk9RUlNWXWVtc4SEa2Vpb3N2aW19jYyD'
                'oKuclnl0cHFvbnN4dXl2eIaRsLO3ubrAtKF+gYePkY6Ki4qLiY'
                'qJi4qLkJKYlZWXmZ5HR0dHR0dHRkdHSEhIR0hKSUhKTUpLTk9T'
                'U05SUlRWWmNxeX2Hkm1obnJydXuDrbC4tr+5mZZ3c3JzdnZ4e3'
                'd5fX+eqb+8vb+8vpmIe4WOkI6MjY2Oj4uKiIqMjpaXnJuUko6Q'
                'R0dHR0dHR0dHSEhISEhISUlJTE9KTE5MT1FOT1FVWmBocnyDmp'
                '5rZmpyf4aussi+vLqkhnJ3bnFvcnl9enp1eYiLtLe8u77Bq56E'
                'gYqOkI2KjY+PkpKOioiLkJOVmqGkmZSOkUdHSEhIR0dISUhISE'
                'hJSElKSk1PSktMT1JPS05QVV5na3B+jaaSaGtwf6Cpw77AuKCS'
                'eHRtcm5xcHF1eXZ3eICdob69uby6tI6JgomPjYyLjI6QkJSSjY'
                'mLj5WWk5qkqJyXkphHR0hISEhISUlJSElJSklJSktNTEpLS0xP'
                'T05RV11pcnJ0hZaZg2l1h5rAwNPFoIt5fHNxbW9ucXJzcHNxdo'
                'aWtbO9ub6+q5SAhYuQjImIioyOkJOUj4yLkZeZlpCTm6SjopiW'
                'R0dHSEhISUlJSUlJSUpJSElKS0tLTUpKTlJVWGxyeHx8fYWNio'
                'B2fqOzzM/JtH1vbnh2cGxscXN0d29zdX+ntL23s7S5tJOJhIuT'
                'jouIiIqNjpCSkI6QkpubmZOOkJadqa2moEdGRkdISEhISEhISE'
                'hJSEhJSUpLTExLTlVUWWl9gIGCgYB+eHWDkpnPycTGnYVraHB0'
                'd3NwcXt8d3lvdYiWvLy6sq64p5yGi5KTmIqLiYmMj5CPjoyOlZ'
                'qhmZaRkpWbmZ2gpahHR0hISEhJSUhHSEhHSkhHSktMT09UU1BW'
                'XHN2kYKOkH2Kdmtxg6m52c+zrYZ9cHJwcnN6dHN+e3F2cn6aqb'
                'q0trO0spaEg5Kclo6Ljo2MjpeTjI+RlZiamJWRj5GTk5ORl6ms'
                'R0dHR0hISEdHR0hISUpHR0pLTExPU1NSW2qJipOPi4l4hXl9jJ'
                'Tc4eLSmqCJhYCBd3V1en58d3dxdn6Tsqu2t7m7sKOBhpGbmpOM'
                'iI6QkJCVlI2Ml5qZlJGRk5SSko+QkJahoEdHR0dHSEdHSEhISE'
                'lKSElKTExLTlBRU2p+lJWNkYuPiJGEjZui5OWwmHuIiYmJinh3'
                'eH1+gHN6fHuKnLWxs7K3upaIgIyZmZKOjYyPj5OWlpKPkpycmJ'
                'OQkZOVkpKQkZGUl5VHR0dHSUpJR0hJSElJSkpLS01PTkxPTleM'
                'kpKYjot1g3+Ulp+5v9HLiHh3f4eDgX90d3yAgIFzeoeJmaOmsb'
                'Wzq6N9gJGSlJONjI6QkJCXl5KRlZ2emJSTlJWUkpGRkpOVlJGN'
                'R0dHR01OSkdISkpLSktKS0tOVlZLT09lpJ+Sm5aNdIyAl6Cd6O'
                'TStnx4fYJ/e3h6dnt9fnx+fYWepqSjoqy1t5eKfI6Wk5COjI+Q'
                'j5CUmpSQlp2dnpWQkpaWlJOQkZKVl5SOiUdISEhKS0lISUpLS0'
                'tKSUtMT1VTSk5UfK2imaKUhWKEg7bt7evr7bZ4fHyFfXx4fHx+'
                'eXd4fo2ZsreppKeqsqqCho+Vk5KQjY2TlZKTk5eVlp6fm5eTkZ'
                'SWlZOUkJOXlpSRjY1HSEhIR0dISEpKS0pKSkpLTU5PTkpQZ4+g'
                'm6CnlY1jjZjb7+3u8PG1en2AiYaHfX19e3Z4f4SWnKisrrGnpZ'
                'aOgpGblZSSkZCQlZiXlJCTmp6emZmSlJWVlJOTlJOWmpaSj4yQ'
                'R0dHR0dHSEhISUpKSUtLTExMTU5LVIeak5CmrJaSXo6r8O/r8P'
                'LjpX5+iIeWloN8enh1f4OKnqSmoK20opZ6f5SamJqXkZGVlZSW'
                'mZKSlpqcm5WTlJeXlZKUlpabmJaTk5GLjUVGRkZHR0dISEhJSk'
                'tNS01LTE1OTFaeo42QoqWNnGiRxu/u7/Hyrpd2fpKfmY99fXp3'
                'hIuNjpuXoKKiqZWIg4uTlpSTk5STmKiglpSWl5mcnZaSkpiYk5'
                'OQlJiUmZiUkZGOjItHR0dHSEhISUhJS0xLTUpLS01NUFFqqKRx'
                'hZ6dd45znOnv7u7vx6ibcnaeqpWHdnh9gY6RkJGZm6SjlpKDho'
                'yTlZaVk5SRkpiin5mSmp6em5eTk5SYl5GRkJWbmpWVk5CNjZCS'
                'SEdHSEhISUpKS0tLS0xKS01PTVFfgqSbYYuhl3OGfJ3u6+7tpo'
                '+jm3Z3n6iOg3V3gIaNjJSYnaGgnIyGf4uYm5WSkpGSkJOUl5ic'
                'mZ6fnJaUk5SWlJORkZaZnZ2TkpGPjI6TmEhHR0hHR0lKSktMTE'
                'pMS01PUEtQbIqbjHCXp5pwgYSh8fDw8X2LqJR6fIuOgoB5fIOJ'
                'kJCSnqCgk42FhoyUnJmRj5CQkJGVkZGSnKKgm5WSk5WVlpGRkZ'
                'SgoJ6blpGPjo2QlZVISEdGRkdJSUhJTE9JTExNTk9LUXKQkHqJ'
                'maOfZYGUwfPz781ykquOeYOGiIWEgYGFiY+Skp2alISChI2ZlZ'
                'ORjo+RkpCUlJGVlpqgnJaSkpaWk5ORkpSYqKacl5SPjo+PkpWR'
                'R0hHRUZGR0hISUpLSUxOTU1PTVd7jHlvmqGhn2KIqu7089+XdZ'
                'OlknmIi4qJiIiKiouKi5iZjYV/g42Wm5KNjY6Qk5WUk5CQnqCd'
                'nZOSkpWYlZKSkpWfpqyll5GOjY+RkZOTkUdIR0VGRkdJSUlJSU'
                'pNUU1MUE9diYplb52noJtgjM709PTuk3qUoJN5gYOChYaLkJKQ'
                'jY6Uj4OAiI2XmZSQjo+PkZOVlpOSlaCfnJmOkJOVlZOSlJafsL'
                'KknZOQjY6RkpOTkpFHR0dGRkhKS0lJS09KTFJOS1BPWY+QbX+Y'
                'o56bX43k9PP08IxzkpiIeHt8gIuMkZSTkY+ShoSBh5eanJWLjZ'
                'CQjo+TlpSUnKCfl5aTkJOVk5KRkpShrbuxl5WSlI+SkpOUk5CP'
                'SEhIR0lKS0lISk1NS1FRUVFQT2SThF1vl6Gjnl+Q4PTy9OqJdY'
                'qBfHx3foOPjpSSk5WRkoF/iJChoZWSkJCPkIyMk5GWnKWhl5WU'
                'kJOUk5OPlJebs7mnnZOWl4+NlZKSj46PlEdISUlJSUlJSEtOT0'
                'pOTk9OT1BqmY9iapmko55dk+7z7fP0kG+Gfnl6e4SIkJKYlZOT'
                'jo+Fh5OWnpiPjJSUjY2PjZGXnZ2dl5GRkpGRk5CRkJWltb2ql5'
                'qWk5KRkZONjouKi5BHSElKSUhISElLTk1KTE9RTE5SbZ6db3Of'
                'qaCXZ5Ly8fDw85ZrhYJ5eoGMj5CSlZSTkoyLh46anJiTj46QlJ'
                'KRkpGVnaWdl5GNj5GSkpaQkJeasrmtmpCYlZCOk5OPiYyMjJCW'
                'SEhJSklISUlJS0tKSUpQU0xOU2ugpHl5prGhlmeO8fHu8fOdcY'
                'WDf4OLkpKUk5SUlpeTkJOZop2Rjo+QkpWYk4+UoaWimpSRkJGS'
                'kpKUkpKkrraxnpiSk4+Qj5GQjoqLio+cpkhISUpKSUpKSUlKSU'
                'lKT1NMTVRroqJ1dKOvo5lglvLz7fHzkXqDgYOMkpKPlJaVlJSV'
                'lpahoqCXi4uRlJSVl5aVnqakmpeVlZaVlJKSkZeasbSmoqKgl5'
                'CLkJCNio2MiouYtsBHSEpLSUhJSklJS0xKS09RS01WdqGac3md'
                'pZuOV4nr8vPzmXl8hIiMjY2OkJKYl5aRkZicpqKXkYuNkpWSkp'
                'Obq6+gn5WWlZeZmJWUkpKirbeolJacnpyUio2Li4mLioyYrLq0'
                'SElLTElISUtKSk1QTk5RUUxPWX2gkXOFm52Uf1NqqLnpznZ6g4'
                'yUk42LjZSRl5aXlpiho6SfkY+Nj5GSlZSWnKinm5yUlpWUlZSS'
                'lJOYq7asnpKVkpSamo2KhoqKi4uRpqiimElKS0xLSktNTEtOUF'
                'FRUlJNUFl9oYpxh5iZloJjbImZi32HgZCTl5WRk5GSi46Lk5mf'
                'pqSYlYqOkpSTlJSXpaedm5eVlJaVkpKQkJOisLOtmZaUk5KPlJ'
                'iQjIWGiIyRnKyen6ZKS0tOTUxLT1BPVVNQT1NTTU5iiJ+Sf46g'
                'oX9yaHiBf3VxeIOPmaOdm6CLiIGGioygpZ6bkI2OkZKXmpiTnK'
                'WimZSSkpeYk5KQkpKZsrernpOSkJKRkJOPjY2Ih4qNo56cn6Ov'
                'S01NT09OTlBPT1BPTE9UUk5QZYOhlXuQoqCFeXJ7f313d4GNnJ'
                'ycnZ6hh4OEiZmdrq+cmJCQkZORk5qboaSimpOSlJeYl5OVkpKd'
                'q7ermpiWk4uIjIyNjYuLi46SnbKuqqqytUxNTk9PT1FRT05MTE'
                '1OT1FPT12MpJd5lKKdjYN+fn17eYCMlp+goKSbl4GDi5KwtLWr'
                'lpOSlJOXlZWdnamkm5aTlJaYlZWSl5eYs7epnJSWlI+Ih4mJh4'
                'iHh4uQl6awr6qmqp9LTE1NTE1SUU5OS01OTk1QUE9bi6abf5Wf'
                'mZGHg4B7e4CMl5yboqOkj4l+h5ecurupnJGRlZiUlZeZpKOlnp'
                'aVl5mXlpOUk5WjsLyvlJOTlo+LiYyKiYWGhoaMj5mira6popmO'
                'S0tMTEpMUVFNTU1PTU9QT1FVYn6lnoWUmZSRhn98fIGMmJ+inq'
                'GYlYKDiZCkpqusm5KUlZWWkpGWnqinoZuXl5qal5WUlZmZs76u'
                'n4yQlJeNioqLi4uGh4mMk5GXnLK0p5yMiE1MTExKTFBRTk5PUU'
                '5OT1BRVWqKpJyGkJKOioN2eoCKlqCko5+diYN8hZuitK2hoJeT'
                'mJaUk5CSn6mjnJeZm5uZmJaVkpWlrLu1mJKKkJqbjYqKiomKiI'
                'iMkZWRl565tZ6ShohPTU1NTE1OUVBPUFBRTU5RUFFolqOThI+N'
                'i4B/fISOlp6kpKCYkYKBhZCstLOsn56ampiVkZCUm6mtmpSSlp'
                'mZlpWWlZWZsburopaQi5GcnY2LjIyJiouMjpOTlJ6lsKeSj4qN'
                'T09PT09OTVFRUFFOT01QUk5RbJSijH6NjYh5foqTmp+jpqKckY'
                'd9hZaeubeopaGhnaGYl46RnqKmoZOWlJeXlJOUlpWgpLW3l5qh'
                'l4yOlJiNjY6OiYqQkZKWlZuho5mTio2Tlk1NTlBRUE1PVVFOVF'
                'JRUFZYWG6LoYyJh4KGhYSNk6KgpqSRlnZ7gpCosLmxo6CkoJ6g'
                'mI+Um6ihnJuWlZaWkpaRkJags7uvoJWWmaGSkZSTjY2OjIeMkJ'
                'CUkp2apJ+MjIuRmZlOTk5QT1BRVFFOUFZSUlVbVVh4ipyJhYaB'
                'hIiJnJ6in6Gjhn52f5CevMGuop+jo5+kn5SUo6inn5mXmpmZlp'
                'CTlJihprium5mXmZSUk5KTkYyOj4uKjpGRlpqmo4+NhoqRmqao'
                'Tk5OT09OUVRPT1NUUVJaXVNYfYeWjISGfX+PlaCen5+amX1yf4'
                'yntNC+nZigpaaipJ6Ynq6tnpmXl5+blpWSkpijsrSuoZOTlJeR'
                'kJWWl5CJjJGQjI+WlZ2kpJ+KhoqRk5q1xU5OT09PTU9RT1JYVF'
                'FUWlhWYIePlI+Cg3t+mKGkoqCkkIR2eo+axc6+qZGbqKSkoZue'
                'o6mspJiXm5yemZOUlpekrbeznaOcl5KUk5STlZWPiYqPkJGWpa'
                'WoqJmSiIiRlJmozc5PT09QTlBRUFBRU1VTWVlWaHeRmJKMfYKH'
                'jZ6lp6igoYd+gpShp9fJp52Ypq6km5iZoauqo5yamZ2gmpeWlZ'
                'yjtrWsn5SfopyRkJCSkpGRkI2NkJGprryzpp+QjIuSmJO227ud'
                'UE9OT1FVVFFRT09RVFxaW4iOlpuSin2KmaCipKeol5GAhJWit8'
                'DPtp2fpq6vqpiWo6qnppiXnZucnpeZmJupsrmwn5eQlJeblZGN'
                'j5CQj4+OkJSbyMS6p5aRjIySn6Sp7OeXjU9OTk9VVlNRUE9OT1'
                'VZZXaZlZqZkIuNl6KnoqKnpYqFhoyepdPatamho6mrrKueoKyp'
                'oqCVmJybmpqZmp6ls7aso5WUkZGQlpiYkZCMjYyPl6Sxvbqzn5'
                'eNi46RlJrH4+O5kZVOTU9RVVNSUE5OUFJVWHSQn5qcl4uOmJ2i'
                'pJ+impV7hZicrLfZ2KSkqKOkqKempa6poaCbmJ2bnJqZnJiqr7'
                'OvoJyPjZGSjo+RmJOQj4yLmMbu7+6joJOTjYySlpib7/CilZKX'
                'T09PUVNSU1FPU1ddZoCWnqKfm5uTlqOhnJ2knJOJfpKcm7TB1r'
                'akr7Wtn6OlpK2wop2ZnZqdmpaVk5yasrGnppqUjoyPkZGQkZGQ'
                'k5CQkJnv7+nYoJ6Pj5CWmZfD3e26jJOSklBRUVJUUVJSUlNZZH'
                '+VoKKgn52anJyhoaCcmph/iZCeoqbT3LiqsMPEspucpaepp5qZ'
                'maCbnZqWl5mnq7CvoZySj4yKiYyOjo6OjpCQk5+r17ytppyYjI'
                '+Wl5+s8fGik4+QkpNPUlNVVVNVU1RaaX+cnJ+dmZydmp6ioqGi'
                'o5KJdIqYnqay6+qmnrG3sameoqisop+Zm56imZmYl5ygr7WrqZ'
                'uUjIuJh4iLjY6OjYyNjJKsta6inZ6TkY6Wm5e46fDgjY2TkpGS'
                'UFFUWlpgYlVbc4qUnJidoZyYmpmfpKWknp+Cc4uXm6C40Ovrmp'
                'q2tqqnp6mpqJmbnJ+fnpqYmJumqK2tpJ+TkYyLiIeMjY2Nj46M'
                'jI+XuLegn5WXkJGVmZ2q7fGum46RkpWRkFVVVmBkaXNjhJaYlZ'
                'KXoqeemZ2hnp2fpJKLfYabn5yk5evr6paetbWrqq+noZ6TmZ6e'
                'm5mamqKnsa+uqqKZjY2LjIqKjI2MjIyMjI2Wn7S3oZyOkZKXl5'
                'm/7O67kI2Rk4+UkpFhXlxgYWF3hqKjmpOLmaKjnp+ipJydmZp8'
                'f4+coqGcpOvr6+qXnqmrqq2to5qalpydnZmam5+usbWwrKulnI'
                '2MiYqLjIqMjYyMjI+VnJ+lrp6WjZWXm5628fGkk42RkJCRk5OR'
                'Z2VhY2JqipiuppyWi5mgoaOlop+go5OJfY2boKajo6jr6+vqlp'
                '2mrLCyo5+ampyemp2cnaGouri4sKijoaKVkImKjI2MjY2Oj5GV'
                'm6ekn5+UkpGam5/H7PC7jY+PlI+QlZKTkmdoZ25php2cqKKbl5'
                'ecnp+ioaKkn594epKboqWloqer5OnozpecpqipqpienJyenZef'
                'nZ+xtMO+vrakm5ehnZWMjY2Njo6Nj5KWmJylo5qWj5Sam6Gr8v'
                'CnmY+SkZGQkpOSlJZohHqLmZqnoqScmpman6KhoaGgpZyQgYOa'
                'n6KkpqavrrzAzbKXn6iqo6GYm5ufmp2cmaGouru9xcavnZyTmJ'
                '6clZGQj4uJj5GUlJyfm5uNi46UmJq5zPHEkJCPkZGSk5OSkY+Q'
                'g4RxlKe89/WfoJebnZ2gn6GkoJ6Lg4ySm56joqWmq6u6t7mqnq'
                'eqpp6emp2goJean6CssLm+v8K+rJ2akZKUlZKQj42LjJKVmpyj'
                'pJSTio2Vlpup8vKsmJGQkpOTk5OSkpGQkoqSmq/39vb3y5ianJ'
                '6cnpybnp2Ze4KXnJqcoqCho6yuvbWzrKOqo5+bnZ6fnZ6anaSn'
                'tr28vLm2squgmJGQkZKSk5KPjZGXl5mZm5yQkpGYnJq72fLeko'
                '+Uk5OUk5KRkJGRkZKUtPX39vb397eglJ2bmpyanZ6YlomQnJyb'
                'nKGfmJyts7Grraylp52dnZ+gn5qfoqWurba5vbKqpqeppJuRj4'
                '+Pj5GSj5WZmZmcmZSVi5Canpuo8vKvmJKSk5WTk5KRkJCQkJCS'
                'rur39vbx9/fApJucmZqZnqukk5KYnJ2anZ+hoZedt7yoo6ippa'
                'WcoaGgn5+eoqeruLS1r7WpoJ6eoJ6bkpCPjYuOkZKbmpeboJqR'
                'j5GUmZqv2/Ljk46WlZGTkpKQkJCRkZGRk/P49/b19Pb386Scnp'
                'qcnKCnno6TnJ+dmaCioqSgpcjBpqOpqZ6gn6Sin52hn6Osq7i0'
                'uLSrpZ+cmpiWl5aTkY2LjZOWmJqanZeRjY+ZmZai8PK6nI+RlJ'
                'OTkpKQkJCRkpKRkpf4+PT29fb29ff3kY2bnKCemZKOlpmbnpyf'
                'n6Ojpau6raOloqKanZ6hnp6coqCmtbC7t7y8pqWenJyYk5KSkZ'
                'COi42RlJmcnpuPjZCVlJelyPLklZCQlZOSlZKSkZCRkZKSkpWj'
                '+Pj39/b19Pb29ZuTmpifnpOTlpiWlqCgm5ujoqetrJ+gpZubm5'
                '2enpufnKGnq7u2ycO+tqamnZqfnJSRkJCQkI6QkZGXl5SSi5CW'
                'm5qj8PLDpI+Rk5WXk5SSk5KRkpKSkpOcsvf29vT29vf39/e+lJ'
                'Kco5+bmpiRlp2bmpyeqqSqqaGgnZ2bmp2fnJ2dnaaoqK23u8/Q'
                'sauho6CbnZ6PlJOTlI+Qj5COkZOMjY2XmpijvvLumJSRkpSVlp'
                'WTk5STlJGUlpabs7b39/b29vf39vf3tZOSmZ+enpmSkpidmZme'
                'oa+sqaaenZiYmZybnZqam56jqKyrws7LuqWhn6KfnKCej5KQk5'
                'aTk5SVko6Si5GWl5ae7fLbpI+SlZOVlZWVlZSUlJKTlZWhrbay'
                '8/f29/X29/b396malJifoqCXiI+bnZeXoKGqqqShnpyXmZmamJ'
                'mYmJ6joaWxteHnx6+hnJ+hnp2gn5CQkpecmZqamZWOko+Wmpeh'
                'uPLynpSSkpWTlpWVlZWUk5OTmJeXs7m1tdL09/b39vf39/WnnZ'
                'WYpqiakYSOmpuWl6Gho6Sgn52bmJ2bmZmYmZukp5+irr7l4bGo'
                'oJ2foZ6dnpyRk5abn56dnJmUjpOXmJae6PLUs4+SlpOVl5WVlZ'
                'WUlJOSlZuepL23r7Wy4/f39/b39+W8oJeZnKqmioaIkJeXlZik'
                'o6Olop+enJibmJqamZ+kqqigpKy6yLeho52en6KenJmXkpeam5'
                'yenJmWlJOWnZmcufDwnJmRlJWWl5eUlpaUlJWVk5ifsbe9s7K0'
                'qLvv9/f39/TJuKaVo6alm3x/ipGUlJaZoqOjpaOgpKegm5KanJ'
                '6mq6eio6q7v7+3qKWZm6GinpuWlJSZn52anJmXlJWZl5ic2vDR'
                'rpGTlZiWl5WUlZaWlJOUlZWjrry4s7K4ta61yOf39/b19vfMnq'
                'yzmI16gIuQk5OYmp+foKCjpayxsaaUmJ2gpaadnaeqwMPO07Oo'
                'mJqlpZ2cl5WVmZ+empmVlZWWl5agtfLypZeTk5SWlZSTlZWVlZ'
                'WTlJiftre5sLCxtrG5w66vscnw9fb3uKipsoeAf4WKj5KTm5ye'
                'nZ2cpayur7q5mpmfoJ6dl5uqpq250tapoZiaqaeenpqYlpianJ'
                'iWkpSVmZeh6/Det5ORk5WTlJeVk5OUk5SWlpeitru2srG6s6mm'
                's8bDs6nX9vb29+2yopiHgYCBiIqRkJ6joZ2Sk6CorLHL0a2foq'
                'OanZ6hu725prC8nZaXnaylnJ2am5iWmZyTk5KWl5Siv+/yoZWV'
                'lJKTlZSUlJOWl5aTlJWeubq3tLK7uLWmoqu5sbbJ8fb29vfz25'
                'aOgH6AhIuOkpWgnZaVjY+Yoaq209m2p52emJyfqd3n47SZo5mV'
                'mZ+sopqbmpuXlpqVkpKVmJel7PHou5KRlJaUlZeVlZaUlpaUk5'
                'WisLi2tbe2ua+mmpixvK/f9fX19fb399mRioGBhYiNjZKapJqR'
                'ko2PkZacp73DrKWWm5mfobDo6uu8jJiVlKCop52am5mclpeWk5'
                'SWmJmn0/HtpJSQlJWXlpaXlZaXlZaWlpactr60s7O4t7OjnpiY'
                'srWq6PX19fb39/PbjYqEh4uJi4iWm56VkZORk5CPkpmkppyalp'
                'ydoqOz5ejUq42ZlZSmraCbnJ2ampSWkpWWmZmg8PHRqpKSlZiV'
                'lJiXl5WWmJaXmJqkrbq7sbW1uLCmm5ybnKzAyvLz9fX29/fnzI'
                'iHgYSHhoiHmZmXkpWVkZKSkpWYm5uWlpqdnJ2guOjpuZuTm5eW'
                'paSbm6GhmpiSlJSYl5quzfHqnZWRlZaVlZaXlpaVl5iWlpyftL'
                'mytLK2tLKlnJicnZ6n3vP19fX39/Px5rWDgn5/gICEh5malpee'
                'mpCNjpScmpWUlZibnZiXoMLn6LGRlJmYm6Gbm52ioZmWkpSWmZ'
                '6p8fGyoJOSk5aUlJiYlJWWlZaXlZanqLi6sbO3s6mmnpuZnp6e'
                'ruX19ff39+zK3+O2foB7fHx+hImWmpaYoJuTkZOan5aLjZKXmp'
                '2VkqDJ5eSpjJGYmZ+dnJ2fnpyXl5aYl53D5PHdl5SUlZWVlJSV'
                'k5OUlZSVl5ebsrW1tbS1tKudnZqbmp2dnc/u9vf397qP3fbQrn'
                'h6eHt8f4aKlJiQjZWWmZyhoqCPgomSlZueko2gzefno4yRmpme'
                'mJ+goZuZl5mcnay+8fGtl5OTlZeUlJWUlJSTlJWUlJicqbS8sa'
                '20sKmgmZqanJqbmpri8vP394eTvO/z1K13dHZ7d3qFjJWPgIyY'
                'm5eep5+WkYiJkpKamZKQlbLn5LGTlZmdmZido52WmpaanK/t8O'
                '7Hk5SUlZWWlpWVlZWVlZeYlZifrrOysrCzrayhmpqam5ycnJ2g'
                '8fT395+Y8ff39duodXR3eXZ4gYiYlYiNjYuOm6Wdk46Kio6OkZ'
                'iWj4+n6OqumZqdnZuZm52fk5aXmsfp8OWil5KXlJSUlZaWlZWU'
                'lJmXlpahrLa0sq+wtqqhm5mbnJydnJ2gofb29/aVv/f39/fpn3'
                'N2d3d3eX+GmpiKi4mDhJKnn5CMkI+Ojo+Wl5SXs+PhpZ6hoJue'
                'naGgm5CUqrzw7r2mk5GVmZWWlZaWlZWVlZWYlpiesba3srGwrr'
                'KjnJqbmp2dnJueoKD39/XHn9z39/b24p1yd3h2dnl/iZmWiYmE'
                'f4OUs6mRjJaWkZKQk5OWoLO1pZ6gpKGXm5+popuirO7w1bGUlZ'
                'WWl5aVl5aWlZWVlZWWlZeisLmzs7Oysqukm5ycnJucnJycnZyc'
                '3tSVm5ve9/f39t2acXd4dnZ6gIqXloaDe3iFn8u5k4qZm5aYk5'
                'WWmJ2fmJadoqOgmJyepqmz7+/u05uYlJWYl5SUl5aVlZWVlZSU'
                'lpifsbm5rrC2s66inZmbm5ucm5udnZybmn2OkJye2/X19/ffmn'
                'J2eHd3eoOKko+Af3h2gZrWx5SFlJuYmpmXl5aXlZKan6Kfnp6k'
                'pKLY6e/VpZmSlZeWlpaUl5mVlJWWlpWVlZaprre2tK6zt6yhm5'
                'yZmpqcnJubnZ2cmpl6kYeNkdH29vb20Zt0dnl6eHuDiI6Lfn97'
                'd3+V1MuVgo+Yl5ialpKQl5aRmZyem5miq6uizcmnmpOTlZaVlZ'
                'WXlpiXlZSVlpaWl5ugt7e2sLSysKufmpiamZqam5ubm5ubm5qZ'
                'dIV1iZTD9fb298ycdnZ7fXl8hIeNj4ODe3eCntLMlYGLlZaWmp'
                'eUkpybkJWXm5iTnqitoZibl5OVmpiVlZiVlpaWk5WWlpaWl5um'
                's7e3trC0s6ygm5qZmZiampmam5uamZubmnJ8cnqLzPX29ve8jH'
                'V7e3t9f4aLkYqJhH16haTaz5iCiZOWnKqclJWYlpKSkZOUj5Wc'
                'oZ2PkZaal5aXlpaWlpaVlZWWlZWVlZmgsLm5tbG5ta+jn5ycmZ'
                'mbm5mamZmZmJiZmZhxenJ+jc/29vb3v413e3h4e36FiJmRkIyG'
                'gYie49iXf4WSmaOhl5OWlZGRlZCSko6TmJuWlZOUlZOUlZSWlp'
                'WVlZSVlpWUlZehqrW1tLSxtKulnJuampqanJuamZqZmZmZmpqa'
                'cXhygo7S9vf298OPen56eXt+h4yfmZiTjouSpOril3yCjpmjmo'
                '+LkZOQjpKTkpCOlpydmJmVk5OSk5WVlZWVlZWUlZWVlZmerLK1'
                's7Kzr6ygnZmZm5ubm5uampmZmZmampqam3B2cICR0vX29vfCjX'
                'qCgX58foiQpaWkmZKRmbLr5ZZ9gImQlY+IhYqOi4mJlpWUk5yl'
                'rKudmpiXlpaWlpWVlZaWlpWVlJijqrKusrC1s6mjnJyam5qanZ'
                '6bmpqZmZqampucnJxvdG15lNT19PX3vYx6g4OBfn6Hjp2hpZuV'
                'lJuu6OWTfH+FiIeChIWHhoWFho+TmJecoLC9qaSenZ6dmZeWlp'
                'aXl5aVlZmfrrCzrbCxsaugnJqbmpqYmZ+inZuamZmampucnp6c'
                'b3VteJzb9fT197uKe4KBgYGBh4uWm6iprrGvv+rnkXp9goSBfo'
                'CCgoKFhoaDjJiXlpSfqbConp2io5yamJeXl5aVlpipqrKxtbKz'
                's6ahmpqZm5mZmpygoJyamZmampubm52cmm95cX2n5Pb19vfAi3'
                'uAgYGBgYiMmKGwt7/HyNrt7JF5e3+EgIB+fn6BhYeIgoaRlpiS'
                'lpefoJyeoqCcnZ2amJaVlpugtbGysLazsaqdm5iamZqZmaOinp'
                'uamZmampuam5ubmplwe3SDsev39/f4xo15gIGAfn+Ij5efp6Kd'
                'naO78O+SeXp+hYGAfn5+foGIjoiGjJainpeRkJqho6KbmJ2gnZ'
                'iVlZiirbSwsK+0rqefmpiYmpqampumo5yampqam5uam5uamZiY'
                'cn13hb71+Pj4+LmIeoCFgXx7gouYqJqQhoaOrvHxlnt9gIWIg4'
                'KBgIGEh4uFh4yYoqOcl42Ql56ipKKeoaSbl5ehsrK0srS0sKmc'
                'npqbnJybmZueo5yZmZmampqanJ2cmpmYmHV/eIbG9vj4+Pizhn'
                '+BgYB9fIGJkZmOiYKDjq3x8Zd8foGEg3+AgICBg4WHiImMk5iZ'
                'mZuWj42Sl56ipaKlo6irsbavtbW2sqqimpydm5ucnJucnp2bmZ'
                'mZmpqampycmpmYmJl7hXqHwvX4+Pj4t4iBgYKDgX6BiIqMiIeD'
                'hY+08vCTe36AgX59fn+AgYKDhImJio6OjZOaoZuTjY6WnqCfpa'
                'u0s7O1s7Gzs6qgm5iam5mZmpycm5ubmpqamZmZmpubmpmZmJiZ'
                'eoV7ir/z9/f4+LyKgIOIiIOAgYaGh4iKiIeTuPTwkHp9f399fX'
                '5+fn+BgoSGhoiKiomNlZqhoZeOkpmbnqSsr6uqra6orKifmpiY'
                'mpmZmZucm5qamZqampqZmpqampmZmZmZmHN+eYjU9vf3+Pi1iI'
                'CFiYeDgIGCg4SJjIiFkLHz8JB7foGAfn5+f39/gYKDhISFiIqJ'
                'i4+XmqOon5SSlaOlqaqnqKahpqagmpiYmJqYmZycnJqZmZmZmZ'
                'qZmpqbmZmYmJmZmJhye3WD5Pf3+Pj4sIZ/goOCgYKCgYSEioqF'
                'go+u8O2PfYGDgoB+f3+AgIGBgoSDhIeIiomLlJScqaqjm5agoa'
                'inpKalo6SimpiYmJiZmJuenZuZmZmZmpqamZmZmZiZmZiYmJiZ'
                'eH51hOz49/j4+K6Gfn9/f3+AgYKEhIqJhoOTs/DsjX2ChIOAfn'
                '+AgYCBgIGEhIWGiIiIiIuOk5qiq6qimZunqKSjoqaenJeXmZeY'
                'mJqdn52ampqampqampmZmJiYmJiYmJiYmXp/d43y+Pj4+Piphn'
                '1/gIB+fH6CgoOJiYeFmLf18I59g4WEgYCAgIGAgIGBg4aIiIiI'
                'h4aLjY+RkZaepJmXo6uppZycm5mVl5qYmZmZm5uamZmampmam5'
                'uamZmYmJiYmJmZmJh4gXeH9Pj4+Pj4nIF+f35/fHt+goKChYqH'
                'h5rU9POLe4GGgX+DgoGBgYKBgYGChIeHhoaIioyOjo6PkpSUmK'
                'SmqKOdmpeWlpiYmJmZmJiYmJiZmpuam5qZmZmYmZiYmJiYmJiY'
                'dHt1iPb4+fj4+JuBfIB/fnx8fn+BhYiHhIeZvvTzinp/hYF/go'
                'GAf3+BgYGCg4SHiIaFh4iJi42Ojo+Pkpaho6akoJ2YlpeYmJiY'
                'mZiYmJiYmZmamZmZmZmYmJiYmJiYmJiYmXN5dov1+fn4+PibgX'
                'uAf318foGAgoaKh4OGlrn09Il5f4SAfoCAgH5+f4CBgoOFiIeG'
                'hYeIiImMj5COjY+SnaCko6KempeXmJiXl5eYmJiYmJiZmZmZmZ'
                'mZmJiYmJiYmJeYmZl1e3eL8vj5+Pj4oYN9gX99fH6DgoWGiYuG'
                'hJLH9fWGeX+Cf35/f39+f39/gIKEhYeHhYWGiIiJjI+Qj46Oj5'
                'meoaKin5qWlpeXl5aWmJiYmJmZmJiYmJiYmJiYmJiYmJiYmZmY'
                'd3x4jen3+Pj4+KKEgYKAf35/goOHhYiMh4OR0PX1hHmAgH5+f3'
                '9+fn5/f3+Cg4SGhoWFhYeHiIqNj4+QjY6Zn6KjpKKXlZWWl5eX'
                'lpeXmJiZmZmZmJiXl5iYmJiYmJiYmZmZmHd7eJDo9/j4+PifhI'
                'GBgIGAf4KDhYaIiYWFlsv19YJ6gH99foB+fX19f4CAgYKDhIWE'
                'hIWGh4iJi42QkY2NmJ6hoqWhmJaVl5eXl5aXl5iYmZmZmZiYl5'
                'eYmJiYmJiYmZiYmJh5fHmR7ff49/f4m4N/gH+BgYGDhYSGh4eE'
                'h5jC9faCeoF/fX5/f319fn+AgYCBgoOEhISEhoeIiYuNj5GNjJ'
                'eenqCjoJqXl5iXl5eXmJiYmJiYmJiYmJiYmJiYmJiYmZmYmJiY'
                'e355jfD3+Pf395yDgIB/gIGBg4WGhoeHhIaXxvX2g3qCgX5+f3'
                '9+fn5/gIGAgIGCg4ODhIWGh4mKjI6PkI2YnZ2fop+Zl5aYmJiY'
                'mJiYmJiYmJiYmJiYmJiYmJiYmJiZmJiYmHp7epD1+Pn4+PiYg3'
                '5/goF+goeGh4qJiYeIpej39oN4gIN/fn+Efn1/fn6Af4GDg4OF'
                'hoeHiIiJioqMjYuNmJ6hn6Kfl5SUlpaWl5eYmJiXl5eYmJiYmJ'
                'iXmJiYmJiZmJmYmJh3f3yU9vj5+Pj4mIN/gIKBf4GEhIWIh4eF'
                'hZjX9/aCeYGEgH+Ag399f35+gICBg4ODhYaGh4eIiouLjIyLjJ'
                'adoJ6hn5eUk5WWlpeYmJiYmJeYmJiXl5iYl5eYmJeYmJiYmJiY'
                'd4N7lfb4+fj4+JWCf4KCgH+AgoKFh4aGhISSwff0gHiBhIB/gI'
                'F+fX5+f4CCgoKCg4WGhYaGh4mLjIuLioyUm5+doJ6YlJOVlpeX'
                'l5eXmJeXl5eXl5eXl5eXl5eXmJiYmJmam3eAe5r3+Pj4+PiTgn'
                '+CgoCAgYGChIaFhYSDkb338392f4F/fn5+fX19f4CBgoOCgoOF'
                'hYSEhYaIiouLi4qMlJuenaGfl5SUlpaWlpaWlpeXl5eXl5eXl5'
                'eXl5eXl5eXmJmam5s=')
        image = np.fromstring(base64.b64decode(data), dtype=np.uint8)
        image = image.astype(float) / 255.
        image.shape = (125,100)
        expected = np.fromstring(base64.b64decode(fdata), dtype=np.uint8)
        expected = expected.astype(float) / 255.
        expected.shape = (125,100)
        result = F.bilateral_filter(image, np.ones(image.shape, bool),
                                    10, .1, 20, .2)
        self.assertTrue(np.max(np.abs(result-expected)) < .1)

class TestLaplacianOfGaussian(unittest.TestCase):
    def test_00_00_zeros(self):
        result = F.laplacian_of_gaussian(np.zeros((10,10)), None, 9, 3)
        self.assertTrue(np.all(result==0))
    
    def test_00_01_zeros_mask(self):
        result = F.laplacian_of_gaussian(np.zeros((10,10)), 
                                         np.zeros((10,10),bool), 9, 3)
        self.assertTrue(np.all(result==0))

    def test_01_01_ring(self):
        '''The LoG should have its lowest value in the center of the ring'''
        i,j = np.mgrid[-20:21,-20:21].astype(float)
        # A ring of radius 3, more or less
        image = (np.abs(i**2+j**2 - 3) < 2).astype(float)
        result = F.laplacian_of_gaussian(image, None, 9, 3)
        self.assertTrue((np.argmin(result) % 41, int(np.argmin(result)/41)) ==
                        (20,20))

class TestCanny(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Test that the Canny filter finds no points for a blank field'''
        result = F.canny(np.zeros((20,20)),np.ones((20,20),bool), 4, 0, 0)
        self.assertFalse(np.any(result))
    
    def test_00_01_zeros_mask(self):
        '''Test that the Canny filter finds no points in a masked image'''
        result = F.canny(np.random.uniform(size=(20,20)),np.zeros((20,20),bool),
                         4,0,0)
        self.assertFalse(np.any(result))
    
    def test_01_01_circle(self):
        '''Test that the Canny filter finds the outlines of a circle'''
        i,j = np.mgrid[-200:200,-200:200].astype(float) / 200
        c = np.abs(np.sqrt(i*i+j*j) - .5) < .02
        result = F.canny(c.astype(float),np.ones(c.shape,bool), 4, 0, 0)
        #
        # erode and dilate the circle to get rings that should contain the
        # outlines
        #
        cd = binary_dilation(c, iterations=3)
        ce = binary_erosion(c,iterations=3)
        cde = np.logical_and(cd, np.logical_not(ce))
        self.assertTrue(np.all(cde[result]))
        #
        # The circle has a radius of 100. There are two rings here, one
        # for the inside edge and one for the outside. So that's 100 * 2 * 2 * 3
        # for those places where pi is still 3. The edge contains both pixels
        # if there's a tie, so we bump the count a little.
        #
        point_count = np.sum(result)
        self.assertTrue(point_count > 1200)
        self.assertTrue(point_count < 1600)
    
    def test_01_02_circle_with_noise(self):
        '''Test that the Canny filter finds the circle outlines in a noisy image'''
        np.random.seed(0)
        i,j = np.mgrid[-200:200,-200:200].astype(float) / 200
        c = np.abs(np.sqrt(i*i+j*j) - .5) < .02
        cf = c.astype(float) * .5 + np.random.uniform(size=c.shape)*.5
        result = F.canny(cf,np.ones(c.shape,bool), 4, .1, .2)
        #
        # erode and dilate the circle to get rings that should contain the
        # outlines
        #
        cd = binary_dilation(c, iterations=4)
        ce = binary_erosion(c,iterations=4)
        cde = np.logical_and(cd, np.logical_not(ce))
        self.assertTrue(np.all(cde[result]))
        point_count = np.sum(result)
        self.assertTrue(point_count > 1200)
        self.assertTrue(point_count < 1600)

class TestRoberts(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Roberts on an array of all zeros'''
        result = F.roberts(np.zeros((10,10)), np.ones((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_00_01_mask(self):
        '''Roberts on a masked array should be zero'''
        np.random.seed(0)
        result = F.roberts(np.random.uniform(size=(10,10)), 
                           np.zeros((10,10),bool))
        self.assertTrue(np.all(result == 0))
    
    def test_01_01(self):
        '''Roberts on a diagonal edge should recreate the diagonal line'''
        
        i,j = np.mgrid[0:10,0:10]
        image = (i >= j).astype(float)
        result = F.roberts(image)
        #
        # Do something a little sketchy to keep from measuring the points
        # at 0,0 and -1,-1 which are eroded
        #
        i[0,0] = 10000
        i[-1,-1] = 10000
        self.assertTrue(np.all(result[i==j]==1))
        self.assertTrue(np.all(result[np.abs(i-j)>1] == 0))
    
    def test_01_02(self):
        '''Roberts on an anti-diagonal edge should recreate the line'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (i > -j).astype(float)
        result = F.roberts(image)
        i[0,-1] = 10000
        i[-1,0] = 10000
        self.assertTrue(np.all(result[i==-j]==1))
        self.assertTrue(np.all(result[np.abs(i+j)>1] == 0))
    
class TestSobel(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Sobel on an array of all zeros'''
        result = F.sobel(np.zeros((10,10)), np.ones((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_00_01_mask(self):
        '''Sobel on a masked array should be zero'''
        np.random.seed(0)
        result = F.sobel(np.random.uniform(size=(10,10)), 
                         np.zeros((10,10),bool))
        self.assertTrue(np.all(result == 0))

    def test_01_01_horizontal(self):
        '''Sobel on an edge should be a horizontal line'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (i>=0).astype(float)
        result = F.sobel(image)
        # Fudge the eroded points
        i[np.abs(j)==5] = 10000
        self.assertTrue(np.all(result[i==0] == 1))
        self.assertTrue(np.all(result[np.abs(i) > 1] == 0))
    
    def test_01_02_vertical(self):
        '''Sobel on a vertical edge should be a vertical line'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (j>=0).astype(float)
        result = F.sobel(image)
        j[np.abs(i)==5] = 10000
        self.assertTrue(np.all(result[j==0] == 1))
        self.assertTrue(np.all(result[np.abs(j) > 1] == 0))

class TestHSobel(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Horizontal sobel on an array of all zeros'''
        result = F.hsobel(np.zeros((10,10)), np.ones((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_00_01_mask(self):
        '''Horizontal Sobel on a masked array should be zero'''
        np.random.seed(0)
        result = F.hsobel(np.random.uniform(size=(10,10)), 
                          np.zeros((10,10),bool))
        self.assertTrue(np.all(result == 0))

    def test_01_01_horizontal(self):
        '''Horizontal Sobel on an edge should be a horizontal line'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (i>=0).astype(float)
        result = F.hsobel(image)
        # Fudge the eroded points
        i[np.abs(j)==5] = 10000
        self.assertTrue(np.all(result[i==0] == 1))
        self.assertTrue(np.all(result[np.abs(i) > 1] == 0))
    
    def test_01_02_vertical(self):
        '''Horizontal Sobel on a vertical edge should be zero'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (j>=0).astype(float)
        result = F.hsobel(image)
        self.assertTrue(np.all(result == 0))

class TestVSobel(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Vertical sobel on an array of all zeros'''
        result = F.vsobel(np.zeros((10,10)), np.ones((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_00_01_mask(self):
        '''Vertical Sobel on a masked array should be zero'''
        np.random.seed(0)
        result = F.vsobel(np.random.uniform(size=(10,10)), 
                          np.zeros((10,10),bool))
        self.assertTrue(np.all(result == 0))

    def test_01_01_vertical(self):
        '''Vertical Sobel on an edge should be a vertical line'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (j>=0).astype(float)
        result = F.vsobel(image)
        # Fudge the eroded points
        j[np.abs(i)==5] = 10000
        self.assertTrue(np.all(result[j==0] == 1))
        self.assertTrue(np.all(result[np.abs(j) > 1] == 0))
    
    def test_01_02_horizontal(self):
        '''vertical Sobel on a horizontal edge should be zero'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (i>=0).astype(float)
        result = F.vsobel(image)
        eps = .000001
        self.assertTrue(np.all(np.abs(result) < eps))

class TestPrewitt(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Prewitt on an array of all zeros'''
        result = F.prewitt(np.zeros((10,10)), np.ones((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_00_01_mask(self):
        '''Prewitt on a masked array should be zero'''
        np.random.seed(0)
        result = F.prewitt(np.random.uniform(size=(10,10)), 
                          np.zeros((10,10),bool))
        eps = .000001
        self.assertTrue(np.all(np.abs(result) < eps))

    def test_01_01_horizontal(self):
        '''Prewitt on an edge should be a horizontal line'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (i>=0).astype(float)
        result = F.prewitt(image)
        # Fudge the eroded points
        i[np.abs(j)==5] = 10000
        eps = .000001
        self.assertTrue(np.all(result[i==0] == 1))
        self.assertTrue(np.all(np.abs(result[np.abs(i) > 1]) < eps))
    
    def test_01_02_vertical(self):
        '''Prewitt on a vertical edge should be a vertical line'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (j>=0).astype(float)
        result = F.prewitt(image)
        eps = .000001
        j[np.abs(i)==5] = 10000
        self.assertTrue(np.all(result[j==0] == 1))
        self.assertTrue(np.all(np.abs(result[np.abs(j) > 1]) < eps))

class TestHPrewitt(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Horizontal sobel on an array of all zeros'''
        result = F.hprewitt(np.zeros((10,10)), np.ones((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_00_01_mask(self):
        '''Horizontal prewitt on a masked array should be zero'''
        np.random.seed(0)
        result = F.hprewitt(np.random.uniform(size=(10,10)), 
                          np.zeros((10,10),bool))
        eps = .000001
        self.assertTrue(np.all(np.abs(result) < eps))

    def test_01_01_horizontal(self):
        '''Horizontal prewitt on an edge should be a horizontal line'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (i>=0).astype(float)
        result = F.hprewitt(image)
        # Fudge the eroded points
        i[np.abs(j)==5] = 10000
        eps = .000001
        self.assertTrue(np.all(result[i==0] == 1))
        self.assertTrue(np.all(np.abs(result[np.abs(i) > 1]) < eps))
    
    def test_01_02_vertical(self):
        '''Horizontal prewitt on a vertical edge should be zero'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (j>=0).astype(float)
        result = F.hprewitt(image)
        eps = .000001
        self.assertTrue(np.all(np.abs(result) < eps))

class TestVPrewitt(unittest.TestCase):
    def test_00_00_zeros(self):
        '''Vertical prewitt on an array of all zeros'''
        result = F.vprewitt(np.zeros((10,10)), np.ones((10,10),bool))
        self.assertTrue(np.all(result==0))
    
    def test_00_01_mask(self):
        '''Vertical prewitt on a masked array should be zero'''
        np.random.seed(0)
        result = F.vprewitt(np.random.uniform(size=(10,10)), 
                          np.zeros((10,10),bool))
        self.assertTrue(np.all(result == 0))

    def test_01_01_vertical(self):
        '''Vertical prewitt on an edge should be a vertical line'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (j>=0).astype(float)
        result = F.vprewitt(image)
        # Fudge the eroded points
        j[np.abs(i)==5] = 10000
        self.assertTrue(np.all(result[j==0] == 1))
        eps = .000001
        self.assertTrue(np.all(np.abs(result[np.abs(j) > 1]) < eps))
    
    def test_01_02_horizontal(self):
        '''vertical prewitt on a horizontal edge should be zero'''
        i,j = np.mgrid[-5:6,-5:6]
        image = (i>=0).astype(float)
        result = F.vprewitt(image)
        eps = .000001
        self.assertTrue(np.all(np.abs(result) < eps))

class TestEnhanceDarkHoles(unittest.TestCase):
    def test_00_00_zeros(self):
        result = F.enhance_dark_holes(np.zeros((15,19)),1,5)
        self.assertTrue(np.all(result == 0))
    
    def test_01_01_positive(self):
        '''See if we pick up holes of given sizes'''
        
        i,j = np.mgrid[-25:26,-25:26].astype(float)
        for r in range(5,11):
            image = (np.abs(np.sqrt(i**2+j**2)-r) <= .5).astype(float)
            eimg = F.enhance_dark_holes(image,r-1, r)
            self.assertTrue(np.all(eimg[np.sqrt(i**2+j**2) < r-1] == 1))
            self.assertTrue(np.all(eimg[np.sqrt(i**2+j**2) >= r] == 0))
            
    def test_01_01_negative(self):
        '''See if we miss holes of the wrong size'''
        i,j = np.mgrid[-25:26,-25:26].astype(float)
        for r in range(5,11):
            image = (np.abs(np.sqrt(i**2+j**2)-r) <= .5).astype(float)
            for lo,hi in ((r-3,r-2),(r+1,r+2)):
                eimg = F.enhance_dark_holes(image,lo, hi)
                self.assertTrue(np.all(eimg==0))
        
class TestKalmanFilter(unittest.TestCase):
    def test_00_00_none(self):
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(kalman_state,
                                 np.zeros(0,int),
                                 np.zeros((0,2)),
                                 np.zeros((0, 4, 4)),
                                 np.zeros((0, 2, 2)))
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 0)
        
    def test_01_01_add_one(self):
        np.random.seed(11)
        locs = np.random.randint(0, 1000, size=(1,2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(kalman_state,
                                 np.ones(1, int) * -1,
                                 locs,
                                 np.zeros((1, 4, 4)),
                                 np.zeros((1, 2, 2)))
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        
    def test_01_02_same_loc_twice(self):
        np.random.seed(12)
        locs = np.random.randint(0, 1000, size=(1,2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(kalman_state,
                                 np.ones(1, int) * -1,
                                 locs,
                                 np.zeros((1, 4, 4)),
                                 np.zeros((1, 2, 2)))
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        
        result = F.kalman_filter(result,
                                 np.zeros(1, int),
                                 locs,
                                 np.eye(4)[np.newaxis, :, :],
                                 np.eye(2)[np.newaxis, :, :])
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))

    def test_01_03_same_loc_thrice(self):
        np.random.seed(13)
        locs = np.random.randint(0, 1000, size=(1,2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(kalman_state,
                                 np.ones(1, int) * -1,
                                 locs,
                                 np.zeros((1,4,4)),
                                 np.zeros((1,2,2)))
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        
        result = F.kalman_filter(result,
                                 np.zeros(1, int),
                                 locs,
                                 np.eye(4)[np.newaxis, :, :],
                                 np.eye(2)[np.newaxis, :, :])
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))
        #
        # The third time through exercises some code to join the state_noise
        #
        result = F.kalman_filter(result,
                                 np.zeros(1, int),
                                 locs,
                                 np.eye(4)[np.newaxis, :, :],
                                 np.eye(2)[np.newaxis, :, :])
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))
    
    def test_01_04_disappear(self):
        np.random.seed(13)
        locs = np.random.randint(0, 1000, size=(1,2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(kalman_state,
                                 np.ones(1, int) * -1,
                                 locs,
                                 np.zeros((1,4,4)),
                                 np.zeros((1,2,2)))
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 1)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        
        result = F.kalman_filter(kalman_state,
                                 np.zeros(0, int),
                                 np.zeros((0,2)),
                                 np.zeros((0,4,4)),
                                 np.zeros((0,2,2)))
        self.assertEqual(len(result.state_vec), 0)
        
    def test_01_05_follow_2(self):
        np.random.seed(15)
        locs = np.random.randint(0, 1000, size=(2,2))
        kalman_state = F.velocity_kalman_model()
        result = F.kalman_filter(kalman_state,
                                 np.ones(2, int) * -1,
                                 locs,
                                 np.zeros((0,2,2)),
                                 np.zeros((0,4,4)))
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 2)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        
        result = F.kalman_filter(result,
                                 np.arange(2),
                                 locs,
                                 np.array([np.eye(4)]*2),
                                 np.array([np.eye(2)]*2))
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 2)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))

        result = F.kalman_filter(result,
                                 np.arange(2),
                                 locs,
                                 np.array([np.eye(4)]*2),
                                 np.array([np.eye(2)]*2))
        self.assertTrue(isinstance(result, F.KalmanState))
        self.assertEqual(len(result.state_vec), 2)
        self.assertTrue(np.all(result.state_vec[:,:2] == locs))
        self.assertTrue(np.all(result.predicted_obs_vec == locs))
        self.assertTrue(np.all(result.noise_var == 0))
        
    def test_01_06_follow_with_movement(self):
        np.random.seed(16)
        vi = 5
        vj = 7
        e = np.ones(2)
        locs = np.random.randint(0, 1000, size=(1,2))
        kalman_state = F.velocity_kalman_model()
        for i in range(100):
            kalman_state = F.kalman_filter(kalman_state,
                                           [0] if i > 0 else [-1],
                                           locs,
                                           np.eye(4)[np.newaxis, :, :] * 2,
                                           np.eye(2)[np.newaxis, :, :] * .5)
            locs[0,0] += vi
            locs[0,1] += vj
            if i > 0:
                new_e = np.abs(kalman_state.predicted_obs_vec[0] - locs[0])
                self.assertTrue(np.all(new_e <= e + np.finfo(np.float32).eps))
                e = new_e

    def test_01_07_scramble_and_follow(self):
        np.random.seed(17)
        nfeatures = 20
        v = np.random.uniform(size=(nfeatures,2)) * 5
        locs = np.random.uniform(size=(nfeatures, 2)) * 100
        e = np.ones((nfeatures, 2))
        q = np.eye(4)[np.newaxis, :, :][np.zeros(nfeatures, int)] * 2
        r = np.eye(2)[np.newaxis, :, :][np.zeros(nfeatures, int)] * .5

        kalman_state = F.kalman_filter(F.velocity_kalman_model(),
                                       -np.ones(nfeatures, int),
                                       locs, q, r)
        locs += v
        for i in range(100):
            scramble = np.random.permutation(np.arange(nfeatures))
            #scramble = np.arange(nfeatures)
            locs = locs[scramble]
            v = v[scramble]
            e = e[scramble]
            kalman_state = F.kalman_filter(kalman_state,
                                           scramble,
                                           locs, q, r)
            locs += v
            new_e = np.abs(kalman_state.predicted_obs_vec - locs)
            self.assertTrue(np.all(new_e <= e + np.finfo(np.float32).eps))
            e = new_e
            
    def test_01_08_scramble_add_and_remove(self):
        np.random.seed(18)
        nfeatures = 20
        v = np.random.uniform(size=(nfeatures,2)) * 5
        locs = np.random.uniform(size=(nfeatures, 2)) * 100
        e = np.ones((nfeatures, 2))
        q = np.eye(4)[np.newaxis, :, :][np.zeros(nfeatures, int)] * 2
        r = np.eye(2)[np.newaxis, :, :][np.zeros(nfeatures, int)] * .5

        kalman_state = F.kalman_filter(F.velocity_kalman_model(),
                                       -np.ones(nfeatures, int),
                                       locs, q, r)
        locs += v
        for i in range(100):
            add = np.random.randint(1, 10)
            remove = np.random.randint(1, nfeatures-1)
            scramble = np.random.permutation(np.arange(nfeatures))[remove:]
            locs = locs[scramble]
            v = v[scramble]
            e = e[scramble]
            new_v = np.random.uniform(size=(add,2)) * 5
            new_locs = np.random.uniform(size=(add, 2)) * 100
            new_e = np.ones((add, 2))
            scramble = np.hstack((scramble, -np.ones(add, int)))
            v = np.vstack((v, new_v))
            locs = np.vstack((locs, new_locs))
            e = np.vstack((e, new_e))
            nfeatures += add - remove
            q = np.eye(4)[np.newaxis, :, :][np.zeros(nfeatures, int)] * 2
            r = np.eye(2)[np.newaxis, :, :][np.zeros(nfeatures, int)] * .5
            kalman_state = F.kalman_filter(kalman_state,
                                           scramble,
                                           locs, q, r)
            locs += v
            new_e = np.abs(kalman_state.predicted_obs_vec - locs)
            self.assertTrue(np.all(new_e[:-add] <= e[:-add] + np.finfo(np.float32).eps))
            e = new_e
            
    def test_02_01_with_noise(self):
        np.random.seed(21)
        nfeatures = 20
        nsteps = 200
        vq = np.random.uniform(size=nfeatures) * 2
        vr = np.random.uniform(size=nfeatures) * .5
        sdq = np.sqrt(vq)
        sdr = np.sqrt(vr)
        v = np.random.uniform(size=(nfeatures,2)) * 10
        locs = np.random.uniform(size=(nfeatures, 2)) * 200
        locs = locs[np.newaxis,:,:] + np.arange(nsteps)[:,np.newaxis, np.newaxis] * v[np.newaxis,:,:]
        process_error = np.random.normal(scale=sdq, size=(nsteps, 2, nfeatures)).transpose((0,2,1))
        measurement_error = np.random.normal(scale=sdr, size=(nsteps, 2, nfeatures)).transpose((0,2,1))
        locs = locs + np.cumsum(process_error, 0)
        meas = locs + measurement_error
        q = np.eye(4)[np.newaxis, :, :][np.zeros(nfeatures, int)] * vq[:, np.newaxis, np.newaxis]
        r = np.eye(2)[np.newaxis, :, :][np.zeros(nfeatures, int)] * vr[:, np.newaxis, np.newaxis]

        obs = np.zeros((nsteps, nfeatures, 2))
        kalman_state = F.kalman_filter(F.velocity_kalman_model(),
                                       -np.ones(nfeatures, int),
                                       meas[0], q, r)
        obs[0] = kalman_state.state_vec[:,:2]
        for i in range(1, nsteps):
            kalman_state = F.kalman_filter(kalman_state,
                                           np.arange(nfeatures),
                                           meas[i], q, r)
            obs[i] = kalman_state.predicted_obs_vec
        #
        # The true variance between the real location and the predicted
        #
        k_var = np.array([np.var(obs[:,i,0] - locs[:,i,0]) for i in range(nfeatures)])
        #
        # I am not sure if the difference between the estimated process
        # variance and the real process variance is reasonable.
        #
        self.assertTrue(np.all(k_var / kalman_state.noise_var[:,0] < 4))
        self.assertTrue(np.all(kalman_state.noise_var[:,0] / k_var < 4))
        
        
class TestPermutations(unittest.TestCase):
    def test_01_01_permute_one(self):
        np.random.seed(11)
        a = [np.random.uniform()]
        b = [p for p in F.permutations(a)]
        self.assertEqual(len(b), 1)
        self.assertEqual(len(b[0]), 1)
        self.assertEqual(b[0][0], a[0])
        
    def test_01_02_permute_two(self):
        np.random.seed(12)
        a = np.random.uniform(size=2)
        b = [p for p in F.permutations(a)]
        self.assertEqual(len(b), 2)
        self.assertEqual(len(b[0]), 2)
        self.assertTrue(np.all(np.array(b) == a[np.array([[0,1],[1,0]])]))
        
    def test_01_03_permute_three(self):
        np.random.seed(13)
        a = np.random.uniform(size=3)
        b = [p for p in F.permutations(a)]
        self.assertEqual(len(b), 6)
        self.assertEqual(len(b[0]), 3)
        expected = np.array([[0,1,2],
                             [0,2,1],
                             [1,0,2],
                             [1,2,0],
                             [2,0,1],
                             [2,1,0]])
        self.assertTrue(np.all(np.array(b) == a[expected]))
       
class TestParity(unittest.TestCase):
    def test_01_01_one(self):
        self.assertEqual(F.parity([1]), 1)
        
    def test_01_02_lots(self):
        np.random.seed(12)
        for i in range(100):
            size = np.random.randint(3,20)
            a = np.arange(size)
            n = np.random.randint(1, 20)
            for j in range(n):
                k,l = np.random.permutation(np.arange(size))[:2]
                a[k],a[l] = a[l],a[k]
            self.assertEqual(F.parity(a), 1 - (n % 2) * 2)
            
class TestDotN(unittest.TestCase):
    def test_00_00_dot_nothing(self):
        result = F.dot_n(np.zeros((0,4,4)), np.zeros((0,4,4)))
        self.assertEqual(len(result), 0)
        
    def test_01_01_dot_2x2(self):
        np.random.seed(11)
        a = np.random.uniform(size = (1,2,2))
        b = np.random.uniform(size = (1,2,2))
        result = F.dot_n(a, b)
        expected = np.array([np.dot(a[0], b[0])])
        np.testing.assert_array_almost_equal(result, expected)

    def test_01_02_dot_2x3(self):
        np.random.seed(12)
        a = np.random.uniform(size = (1,3,2))
        b = np.random.uniform(size = (1,2,3))
        result = F.dot_n(a, b)
        expected = np.array([np.dot(a[0], b[0])])
        np.testing.assert_array_almost_equal(result, expected)

    def test_01_02_dot_nx2x3(self):
        np.random.seed(13)
        a = np.random.uniform(size = (20,3,2))
        b = np.random.uniform(size = (20,2,3))
        result = F.dot_n(a, b)
        expected = np.array([np.dot(a[i], b[i]) for i in range(20)])
        np.testing.assert_array_almost_equal(result, expected)

class TestDetN(unittest.TestCase):
    def test_00_00_det_nothing(self):
        result = F.det_n(np.zeros((0,4,4)))
        self.assertEqual(len(result), 0)
        
    def test_01_01_det_1x1x1(self):
        np.random.seed(11)
        a = np.random.uniform(size=(1,1,1))
        result = F.det_n(a)
        self.assertEqual(len(result), 1)
        self.assertEqual(a[0,0,0], result[0])
        
    def test_01_02_det_1x2x2(self):
        np.random.seed(12)
        a = np.random.uniform(size=(1,2,2))
        result = F.det_n(a)
        expected = np.array([np.linalg.det(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)

    def test_01_03_det_1x3x3(self):
        np.random.seed(13)
        a = np.random.uniform(size=(1,3,3))
        result = F.det_n(a)
        expected = np.array([np.linalg.det(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)

    def test_01_04_det_nx3x3(self):
        np.random.seed(14)
        a = np.random.uniform(size=(21,3,3))
        result = F.det_n(a)
        expected = np.array([np.linalg.det(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)

class TestCofactorN(unittest.TestCase):
    def test_01_01_cofactor_1x2x2(self):
        np.random.seed(11)
        a = np.random.uniform(size=(1,2,2))
        ii, jj = np.mgrid[:(a.shape[1]-1),:(a.shape[1]-1)]
        r = np.arange(a.shape[1])
        for i in range(a.shape[1]):
            for j in range(a.shape[1]):
                result = F.cofactor_n(a, i, j)
                for n in range(a.shape[0]):
                    iii = r[r!=i]
                    jjj = r[r!=j]
                    aa = a[n][iii[ii], jjj[jj]]
                    expected = np.linalg.det(aa)
                    self.assertAlmostEqual(expected, result[n])
                
    def test_01_02_cofactor_1x3x3(self):
        np.random.seed(12)
        a = np.random.uniform(size=(1,3,3))
        ii, jj = np.mgrid[:(a.shape[1]-1),:(a.shape[1]-1)]
        r = np.arange(a.shape[1])
        for i in range(a.shape[1]):
            for j in range(a.shape[1]):
                result = F.cofactor_n(a, i, j)
                for n in range(a.shape[0]):
                    iii = r[r!=i]
                    jjj = r[r!=j]
                    aa = a[n][iii[ii], jjj[jj]]
                    expected = np.linalg.det(aa)
                    self.assertAlmostEqual(expected, result[n])
                
    def test_01_03_cofactor_nx4x4(self):
        np.random.seed(13)
        a = np.random.uniform(size=(21,4,4))
        ii, jj = np.mgrid[:(a.shape[1]-1),:(a.shape[1]-1)]
        r = np.arange(a.shape[1])
        for i in range(a.shape[1]):
            for j in range(a.shape[1]):
                result = F.cofactor_n(a, i, j)
                for n in range(a.shape[0]):
                    iii = r[r!=i]
                    jjj = r[r!=j]
                    aa = a[n][iii[ii], jjj[jj]]
                    expected = np.linalg.det(aa)
                    self.assertAlmostEqual(expected, result[n])
                
class TestInvN(unittest.TestCase):
    def test_01_01_inv_1x1x1(self):
        np.random.seed(11)
        a = np.random.uniform(size=(1,1,1))
        result = F.inv_n(a)
        self.assertEqual(len(result), 1)
        self.assertEqual(a[0,0,0], 1/result[0])
        
    def test_01_02_inv_1x2x2(self):
        np.random.seed(12)
        a = np.random.uniform(size=(1,2,2))
        result = F.inv_n(a)
        expected = np.array([np.linalg.inv(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)

    def test_01_03_inv_1x3x3(self):
        np.random.seed(13)
        a = np.random.uniform(size=(1,3,3))
        result = F.inv_n(a)
        expected = np.array([np.linalg.inv(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)

    def test_01_04_inv_nx3x3(self):
        np.random.seed(14)
        a = np.random.uniform(size=(21,3,3))
        result = F.inv_n(a)
        expected = np.array([np.linalg.inv(a[i]) for i in range(len(a))])
        np.testing.assert_almost_equal(result, expected)
        
class TestConvexHullTransform(unittest.TestCase):
    def test_01_01_zeros(self):
        '''The convex hull transform of an array of identical values is itself'''
        self.assertTrue(np.all(F.convex_hull_transform(np.zeros((10,20))) == 0))
        
    def test_01_02_point(self):
        '''The convex hull transform of 1 foreground pixel is itself'''
        image = np.zeros((10,20))
        image[5,10] = 1
        self.assertTrue(np.all(F.convex_hull_transform(image) == image))
        
    def test_01_03_line(self):
        '''The convex hull transform of a line of foreground pixels is itself'''
        image = np.zeros((10,20))
        image[5, 7:14] = 1
        self.assertTrue(np.all(F.convex_hull_transform(image) == image))

    def test_01_04_convex(self):
        '''The convex hull transform of a convex figure is itself'''
        
        image = np.zeros((10,20))
        image[2:7, 7:14] = 1
        self.assertTrue(np.all(F.convex_hull_transform(image) == image))
        
    def test_01_05_concave(self):
        '''The convex hull transform of a concave figure is the convex hull'''
        expected = np.zeros((10, 20))
        expected[2:8, 7:14] = 1
        image = expected.copy()
        image[4:6, 7:10] = .5
        self.assertTrue(np.all(F.convex_hull_transform(image) == expected))
        
    def test_02_01_two_levels(self):
        '''Test operation on two grayscale levels'''
        
        expected = np.zeros((20, 30))
        expected[3:18, 3:27] = .5
        expected[8:15, 10:20] = 1
        image = expected.copy()
        image[:,15] = 0
        image[10,:] = 0
        # need an odd # of bins in order to have .5 be a bin
        self.assertTrue(np.all(F.convex_hull_transform(image, 7) == expected))
        
    def test_03_01_masked(self):
        '''Test operation on a masked image'''

        expected = np.zeros((20, 30))
        expected[3:18, 3:27] = .5
        expected[8:15, 10:20] = 1
        image = expected.copy()
        image[:,15] = 0
        image[10,:] = 0
        mask = np.ones((20,30), bool)
        mask[:,0] = False
        image[:,0] = .75
        
        result = F.convex_hull_transform(image, levels = 7, mask = mask)
        self.assertTrue(np.all(result == expected))
        
    def test_04_01_many_chunks(self):
        '''Test the two-pass at a single level chunk looping'''
        np.random.seed(41)
        #
        # Make an image that monotonically decreases from the center
        #
        i,j = np.mgrid[-50:51, -50:51].astype(float) / 100.
        image = 1 - np.sqrt(i**2 + j**2)
        expected = image.copy()
        #
        # Riddle it with holes
        #
        holes = np.random.uniform(size=image.shape) < .01
        image[holes] = 0
        result = F.convex_hull_transform(image, levels = 256, chunksize=1000,
                                         pass_cutoff = 256)
        diff = np.abs(result - expected)
        self.assertTrue(np.sum(diff > 1/256.) <= np.sum(holes))
        expected = F.convex_hull_transform(image, pass_cutoff = 256)
        np.testing.assert_equal(result, expected)
    
    def test_04_02_two_pass(self):
        '''Test the two-pass at multiple levels chunk looping'''
        np.random.seed(42)
        #
        # Make an image that monotonically decreases from the center
        #
        i,j = np.mgrid[-50:51, -50:51].astype(float) / 100.
        image = 1 - np.sqrt(i**2 + j**2)
        #
        # Riddle it with holes
        #
        holes = np.random.uniform(size=image.shape) < .01
        image[holes] = 0
        result = F.convex_hull_transform(image, levels = 256, chunksize=1000,
                                         pass_cutoff = 256)
        expected = F.convex_hull_transform(image, pass_cutoff = 256)
        np.testing.assert_equal(result, expected)
        
class TestCircularHough(unittest.TestCase):
    def test_01_01_nothing(self):
        img = np.zeros((10,20))
        result = F.circular_hough(img, 4)
        self.assertTrue(np.all(result == 0))
        
    def test_01_02_circle(self):
        i,j = np.mgrid[-15:16,-15:16]
        circle = np.abs(np.sqrt(i*i+j*j) - 6) <= 1.5
        expected = convolve(circle.astype(float), circle.astype(float)) / np.sum(circle)
        img = F.circular_hough(circle, 6)
        self.assertEqual(img[15,15], 1)
        self.assertTrue(np.all(img[np.abs(np.sqrt(i*i+j*j) - 6) < 1.5] < .25))
        
    def test_01_03_masked(self):
        img = np.zeros((31,62))
        mask = np.ones((31,62), bool)
        i,j = np.mgrid[-15:16,-15:16]
        circle = np.abs(np.sqrt(i*i+j*j) - 6) <= 1.5
        # Do one circle
        img[:,:31] = circle
        # Do a second, but mask it
        img[:,31:] = circle
        mask[:,31:][circle] = False
        result = F.circular_hough(img, 6, mask=mask)
        self.assertEqual(result[15,15], 1)
        self.assertEqual(result[15,15+31], 0)
        
class TestLineIntegration(unittest.TestCase):
    def test_01_01_nothing(self):
        img = np.zeros((23,17))
        result = F.line_integration(img, 0, .95, 2.0)
        np.testing.assert_almost_equal(result, 0)
        
    def test_01_02_two_lines(self):
        img = np.ones((20,30)) * .5
        img[8,10:20] = 1
        img[12,10:20] = 0
        result = F.line_integration(img, 0, 1, 0)
        expected = np.zeros((20,30))
        expected[9:12,10:20] = 1
        expected[8,10:20] = .5
        expected[12,10:20] = .5
        np.testing.assert_almost_equal(result, expected)
        
    def test_01_03_diagonal_lines(self):
        img = np.ones((20,30)) * .5
        i,j = np.mgrid[0:20,0:30]
        img[(i == j-3) & (i <= 15)] = 1
        img[(i == j + 3)] = 0
        expected = np.zeros((20,30), bool)
        expected[(i >= j-3) & (i <= j+3)] = True
        result = F.line_integration(img, -45, 1, 0)
        self.assertTrue(np.mean(result[expected]) > .5)
        self.assertTrue(np.mean(result[~ expected]) < .25)
        
    def test_01_04_decay(self):
        img = np.ones((25,23)) * .5
        img[10,10] = 1
        img[20,10] = 0
        result = F.line_integration(img, 0, .9, 0)
        decay_part = result[11:20,10]
        expected = .9 ** np.arange(1,10) + .9 ** np.arange(9,0,-1)
        expected = ((expected - np.min(expected)) / 
                    (np.max(expected) - np.min(expected)))
        decay_part = ((decay_part - np.min(decay_part)) /
                      (np.max(decay_part) - np.min(decay_part)))
        np.testing.assert_almost_equal(decay_part, expected)
    
    def test_01_05_smooth(self):
        img = np.ones((30,20)) * .5
        img[10,10] = 1
        img[20,10] = 0
        result = F.line_integration(img, 0, 1, .5)
        part = result[15,:]
        part = (part - np.min(part)) / (np.max(part)-np.min(part))
        expected = np.exp(- (np.arange(20)-10)**2 * 2)
        expected = (expected - np.min(expected))/ (np.max(expected) - np.min(expected))
        np.testing.assert_almost_equal(part, expected)

class TestVarianceTransform(unittest.TestCase):
    def test_01_00_zeros(self):
        result = F.variance_transform(np.zeros((30,20)), 1)
        np.testing.assert_almost_equal(result, 0)
        
    def test_01_01_transform(self):
        r = np.random.RandomState()
        r.seed(11)
        img = r.uniform(size=(21,18))
        sigma = 1.5
        result = F.variance_transform(img, sigma)
        #
        # Calculate the variance for one point
        #
        center_i, center_j = 10,9
        i,j = np.mgrid[-center_i:(img.shape[0]-center_i),
                       -center_j:(img.shape[1]-center_j)]
        weight = np.exp(-(i*i+j*j) / (2 * sigma * sigma))
        weight = weight / np.sum(weight)
        mean = np.sum(img * weight)
        norm = img - mean
        var = np.sum(norm * norm * weight)
        self.assertAlmostEqual(var, result[center_i, center_j], 5)
    
    def test_01_02_transform_masked(self):
        r = np.random.RandomState()
        r.seed(12)
        center_i, center_j = 10,9
        img = r.uniform(size=(21,18))
        mask = r.uniform(size=(21,18)) > .25
        mask[center_i, center_j] = True
        sigma = 1.7
        result = F.variance_transform(img, sigma, mask)
        #
        # Calculate the variance for one point
        #
        i,j = np.mgrid[-center_i:(img.shape[0]-center_i),
                       -center_j:(img.shape[1]-center_j)]
        weight = np.exp(-(i*i+j*j) / (2 * sigma * sigma))
        weight[~mask] = 0
        weight = weight / np.sum(weight)
        mean = np.sum(img * weight)
        norm = img - mean
        var = np.sum(norm * norm * weight)
        self.assertAlmostEqual(var, result[center_i, center_j], 5)
        
class TestPoissonEquation(unittest.TestCase):
    def test_00_00_nothing(self):
        image = np.zeros((11, 14), bool)
        p = F.poisson_equation(image)
        np.testing.assert_array_equal(p, 0)
        
    def test_00_01_single(self):
        image = np.zeros((11, 14), bool)
        image[7, 3] = True
        p = F.poisson_equation(image)
        np.testing.assert_array_equal(p[image], 1)
        np.testing.assert_array_equal(p[~image], 0)
        
    def test_01_01_simple(self):
        image = np.array(
            [ [ 0, 0, 0, 0, 0 ],
              [ 0, 0, 1, 0, 0 ],
              [ 0, 1, 1, 1, 0 ],
              [ 0, 0, 1, 0, 0 ],
              [ 0, 0, 0, 0, 0 ] ], bool)
        a = 5. / 3.
        b = a + 1
        self.assertAlmostEqual(b / 4 + 1, a)
        expected = np.array(
            [ [ 0, 0, 0, 0, 0 ],
              [ 0, 0, a, 0, 0 ],
              [ 0, a, b, a, 0 ],
              [ 0, 0, a, 0, 0 ],
              [ 0, 0, 0, 0, 0 ]])
        p = F.poisson_equation(image, convergence=.00001)
        np.testing.assert_almost_equal(p, expected, 4)
        
    def test_01_02_boundary(self):
        # Test an image with pixels at the boundaries.
        image = np.array(
            [ [ 0, 1, 0],
              [ 1, 1, 1],
              [ 0, 1, 0 ]], bool)
        a = 5. / 3.
        b = a + 1
        self.assertAlmostEqual(b / 4 + 1, a)
        expected = np.array(
            [ [ 0, a, 0 ],
              [ a, b, a ],
              [ 0, a, 0 ]])
        p = F.poisson_equation(image, convergence=.00001)
        np.testing.assert_almost_equal(p, expected, 4)
        
    def test_01_03_subsampling(self):
        # Test an image that is large enough to undergo some subsampling
        #
        r = np.random.RandomState()
        r.seed(13)
        image = r.uniform(size=(300, 300)) < .001
        i, j = np.mgrid[-8:9, -8:9]
        kernel = i*i + j*j <=64
        image = binary_dilation(image, kernel)
        p = F.poisson_equation(image, convergence=.001)
        i, j = np.mgrid[0:p.shape[0], 0:p.shape[1]]
        mask = image & (i > 0) & (i < p.shape[0]-1) & (j > 0) & (j < p.shape[1]-1)
        i, j = i[mask], j[mask]
        expected = (p[i+1, j] + p[i-1, j] + p[i, j+1] + p[i, j-1]) / 4 + 1
        np.testing.assert_almost_equal(p[mask], expected, 0)