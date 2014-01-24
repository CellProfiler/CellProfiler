'''test_graytocolor.py - Test the GrayToColor module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''


import numpy as np

import cellprofiler.cpmodule as cpm
import cellprofiler.settings as cps

import base64
import numpy as np
from StringIO import StringIO
import unittest
import zlib

from cellprofiler.preferences import set_headless
set_headless()

import cellprofiler.pipeline as cpp
import cellprofiler.cpmodule as cpm
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw
import cellprofiler.modules.graytocolor as G

OUTPUT_IMAGE_NAME = 'outputimage'
class TestGrayToColor(unittest.TestCase):
    def make_workspace(self, scheme, images, adjustments):
        module = G.GrayToColor()
        module.scheme_choice.value = scheme
        image_names = ["image%d"%i if images[i] is not None
                       else G.LEAVE_THIS_BLACK
                       for i in range(7)]
        for image_name_setting, image_name, adjustment_setting, adjustment\
            in zip((module.red_image_name, module.green_image_name, 
                    module.blue_image_name,
                    module.cyan_image_name, module.magenta_image_name,
                    module.yellow_image_name, module.gray_image_name),
                   image_names,
                   (module.red_adjustment_factor, module.green_adjustment_factor,
                    module.blue_adjustment_factor, module.cyan_adjustment_factor,
                    module.magenta_adjustment_factor, module.yellow_adjustment_factor,
                    module.gray_adjustment_factor),
                   adjustments):
            image_name_setting.value = image_name
            adjustment_setting.value = adjustment
        module.rgb_image_name.value = OUTPUT_IMAGE_NAME
        module.module_num = 1
        pipeline = cpp.Pipeline()
        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.add_module(module)
        image_set_list = cpi.ImageSetList()
        image_set = image_set_list.get_image_set(0)
        for image, image_name in zip(images, image_names):
            if image is not None:
                image_set.add(image_name, cpi.Image(image))
        workspace = cpw.Workspace(pipeline, module, image_set, cpo.ObjectSet(),
                                  cpmeas.Measurements(), image_set_list)
        return workspace, module
        
    def test_01_01_load_matlab(self):
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUvDNz1PwTSxSMDZQMLC0MjW3MjRWMDIwsFQgGTAwevryMzAwrGFi'
                'YKiY8zTCN/+WgYSZ4oKNK3u+hnR69+3uYbwa0jfV6opSaYnUilt+kz15Posk'
                'TeFnsf46/037m032QHj5kZniMeXipDMJs1S1q3rvP6+dp142J4CVYY894wKZ'
                'OPnWWGb+tSJHWZddrmiokbcWXs9SUiDUW8//W7fEUObmC7FbjJ8OrDr62Dbd'
                'p+30zVfLzY9w5VxpDy4ruvHrYElLzPs0GXU3+/1HuVP4P4izXIu2eSs7K5lf'
                'WP1U/4efF0NjHxWXrk7cb+1Vo1Fg9yLRPu3Yyz/FE85cN/8Rlvzs+iI7F+7/'
                'ruWMvvdnON3/41JoxTTHqN35sbS4/5o0M7n+nTVS8QKzd+zVnLPApb/DvfPw'
                'wkIexdWHs8xqbKad3/994d7Uf2J2+9TanGuUPxT90HikOPf+pPP/My6xbqyX'
                'XxO2SS7kYwZXQVe2hasD55vb7N/qTn24sWW/q53ItQKuBYx8CbZbJn+dML3e'
                'O6j8OUPJ5vNbY18+u/63/YNPyMpo0b41dYF/1WNqpcqfn4iVenZpy8Qi2d/f'
                'lPPvlR6zydX1Z65VCn9vsmp6aHpU7NW69KeL1z98mfDokOrsetUT9cc+LjQs'
                '07E5HsaQy3x5+c7jD956Lz/xPfb58rnBq151/1lTUHdpxr3s8y1mSQWbPj2c'
                'f/ic94oQ/zi72w+mWu7ZcbRCavL0srdT/v+RPfZr/1u7HfdDhKezbN7fv/nl'
                'H8XF5098EV7w33E+R/rl8pJDxznkJ2YsC5qeVRv58dtR/5bFp/6vsVj1dWks'
                '/9n9sR9/h389PU87vPXL5+Z6/8l3p19nibJ91upz7fzpD3sTPfTk/tXcbDrP'
                'cC4wzs5dhq93d7dk4V/Fl3o+Kcwfax/8vXLoxY5PWyY93T19r/1WjTwzv/it'
                '589ZGZV6H1117Lro05cXf1z/tf1X/6/zuzL+vnjw/etnmSXn+28rnr//9Kv4'
                'klCPFAAYV4Td')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 4)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, G.GrayToColor))
        self.assertEqual(module.scheme_choice, G.SCHEME_RGB)
        self.assertEqual(module.red_image_name, "Origd0")
        self.assertEqual(module.green_image_name, "Origd2")
        self.assertEqual(module.blue_image_name, "Origd1")
        self.assertEqual(module.rgb_image_name, "ColorImage")
        for adjustment_factor in (module.red_adjustment_factor,
                                  module.green_adjustment_factor,
                                  module.blue_adjustment_factor):
            self.assertEqual(adjustment_factor.value, 1)
    
    def test_01_02_load_v2(self):
        data = ('eJztWUFv2jAUdmiK1k2b2Gm7VPNxmtoooZrUcRlQtg6p0KqgnucSQy2FGBmn'
                'a/cL9hP2M3vccXGWkOABAVPaghJkJc953/v8nr8EjBuV9kmlCj8aJmxU2vtd'
                '4mB45iDepaxfgi7fg0cMI45tSN0SbHsY1nAHmha0zFLxoGQdwqJpfgJqh1Zv'
                'vPJPv98BkPfPz/yWC29th7aWaMJuYc6J2xtuAx28Dfvv/HaBGEGXDr5AjoeH'
                'MUXUX3e7tH07GN1qUNtzcBP1k87+0fT6l5gNT7sRMLx9Rm6w0yI/sZRC5HaO'
                'r8mQUDfEh/Hl3hEv5RJv64r++Mr84Ujxq4h3rlrcn4Hx/qBuL+O6aVLdRB13'
                'E/3C/xuI/fUJdX6d8C+ENnFtck1sDzmQ9FFvNGoRz0yJtzUWbwucH1cDXDkF'
                'V5DGIVob3/D9Lzeow2FflETEOUyJk5fiCPuUkZ5tzjd+bQyvgYMwb2XeIpgr'
                '/+cSXtg1Cl3KoTfEcf2Vx2HNl39uDJ8DTao+/iPqUFYXCorHn6bHN1IcYddw'
                'F3kOh0EoWCMMdzhlt0rzaQE1HRiKuHn5ptX9qfHpYzjdx7n4sfnyEi46ItxO'
                'eF6GT+W5s02Dk+7D8y7zvov0s+p8VXWrqr9N4ltE70rzaE2ex1XzTvueWod8'
                '1+U9ti58SvNYfETdFh9Bt/eQ76br6KH5fi24PruPvB4CVwaz6zjp93+weOwx'
                '6g0y/lXzT1o/x/zQX9LjwTrpbdNxZfC09ZTxz8efFid7LjPcU8KVQabXDLc+'
                'uDLI9Jrh1HF/Ejh5PSbs5P/Swv87mK23D2Bcb8LuYMcZMCr2UZnRDzb7hoZD'
                'kf1v98w48S/riY20eXj2JJ69aTw9hm457YitFuPYv27TYNtlUt12JvAk88/5'
                'n93C7HrLdY7rf/dZhU/P/c/3IgWnhxUL1ttgsfl9P8M/ym0Z/0Xz17Tl84h5'
                '9NGYAIj31Rf1/wuo8TuS')
        pipeline = cpp.Pipeline()
        def callback(caller,event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 2)
        module = pipeline.modules()[1]
        self.assertTrue(isinstance(module, G.GrayToColor))
        self.assertEqual(module.scheme_choice, G.SCHEME_RGB)
        self.assertEqual(module.red_image_name, "Origd0")
        self.assertEqual(module.green_image_name, "Origd2")
        self.assertEqual(module.blue_image_name, "Origd1")
        self.assertEqual(module.rgb_image_name, "ColorImage")
        for adjustment_factor in (module.red_adjustment_factor,
                                  module.green_adjustment_factor,
                                  module.blue_adjustment_factor):
            self.assertEqual(adjustment_factor.value, 1)
        
    def test_02_01_rgb(self):
        np.random.seed(0)
        for combination in ((True, True, True), (True, True, False),
                            (True, False, True), (True, False, False),
                            (False, True, True), (False, True, False),
                            (False, False, True)):
            adjustments = np.random.uniform(size=7)
            images = [np.random.uniform(size=(10,15)) if combination[i] 
                      else None for i in range(3)]
            images += [None] * 4
            workspace, module = self.make_workspace(G.SCHEME_RGB,
                                                    images, adjustments)
            module.run(workspace)
            image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            pixel_data = image.pixel_data
            
            expected = np.dstack([image if image is not None 
                                  else np.zeros((10,15))
                                  for image in images[:3]])
            for i in range(3):
                expected[:,:,i] *= adjustments[i]
            self.assertTrue(np.all(np.abs(expected - pixel_data) <= .00001))
    
    def test_03_01_cmyk(self):
        np.random.seed(0)
        for combination in [[(i & 2^j)!=0 for j in range(4)]
                            for i in range(1,16)]:
            adjustments = np.random.uniform(size=7)
            images = [np.random.uniform(size=(10,15)) if combination[i] 
                      else None for i in range(4)]
            images = [None] * 3 + images
            workspace, module = self.make_workspace(G.SCHEME_CMYK,
                                                    images, adjustments)
            module.run(workspace)
            image = workspace.image_set.get_image(OUTPUT_IMAGE_NAME)
            pixel_data = image.pixel_data
            
            expected = np.array([np.dstack([image * adjustment
                                            if image is not None 
                                            else np.zeros((10,15))]*3) *
                                 np.array(multiplier) / np.sum(multiplier)
                                 for image, multiplier, adjustment in 
                                 ((images[3],(0,1,1), adjustments[3]),
                                  (images[4],(1,1,0), adjustments[4]),
                                  (images[5],(1,0,1), adjustments[5]),
                                  (images[6],(1,1,1), adjustments[6]))])
            expected = np.sum(expected, 0)
            self.assertTrue(np.all(np.abs(expected - pixel_data) <= .00001))
        
