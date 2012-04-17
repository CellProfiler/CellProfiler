'''test_client.py - test the subimager client

'''
# CellProfiler is distributed under the GNU General Public License.
# See the accompanying file LICENSE for details.
# 
# Developed by the Broad Institute
# Copyright 2003-2011
# 
# Please see the AUTHORS file for credits.
# 
# Website: http://www.cellprofiler.org
#
import base64
import numpy as np
import os
import re
import tempfile
import unittest
import urllib
import zlib

import subimager.client as C
import subimager.imagejrequest as IJRQ

root_dir = os.path.split(os.path.split(os.path.split(__file__)[0])[0])[0]
cp_logo_png = os.path.join(root_dir, "cellprofiler", "icons", "CP_logo.png")
cp_logo_url = "file:" + urllib.pathname2url(cp_logo_png)

class TestClient(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        C.start_subimager()
        cls.contextID = None
        cls.modules = None
        
    @classmethod
    def tearDownClass(cls):
        C.stop_subimager()
        
    @classmethod
    def get_context_id(cls):
        if cls.contextID is None:
            xml = IJRQ.RequestType(
                CreateContext = IJRQ.CreateContextRequestType())
            response, _ = C.make_imagej_request(xml, {})
            response = IJRQ.parseString(response)
            cls.contextID = response.CreateContextResponse.ContextID.ContextID
        return cls.contextID
    
    @classmethod
    def get_modules(cls):
        if cls.modules is None:
            xml = IJRQ.RequestType(
                GetModules = IJRQ.GetModulesRequestType(cls.get_context_id()))
            response, _ = C.make_imagej_request(xml, {})
            rm = IJRQ.parseString(response).GetModulesResponse
            assert isinstance(rm, IJRQ.GetModulesResponseType)
            cls.modules = rm.Module
        return cls.modules
    
    def test_01_01_load_color(self):
        global cp_logo_url, cp_logo
        img = C.get_image(cp_logo_url)
        np.testing.assert_almost_equal(cp_logo, img)
        
    def test_01_02_load_by_channel(self):
        global cp_logo_url, cp_logo
        for i in range(cp_logo.shape[2]):
            img = C.get_image(cp_logo_url, channel = i)
            np.testing.assert_almost_equal(cp_logo[:,:,i], img)
            
    def test_02_01_load_xml(self):
        xml = C.get_metadata(cp_logo_url)
        pattern = "<Pixels [^>]+?>"
        pixels = re.search(pattern, xml).group(0)
        for kwd,value in (
            ("SizeX","187"),
            ("SizeY","70"),
            ("SizeZ","1"),
            ("SizeT","1"),
            ("SizeC","4"),
            ("Type","uint8")):
            pattern = '%s="([^"]+)"' % kwd
            m = re.search(pattern, pixels)
            self.assertEqual(m.groups()[0], value)
            
    def test_03_01_save_monochrome(self):
        global cp_logo
        xml = make_xml(size_c=1, samples_per_pixel=1)
        fd, name = tempfile.mkstemp(suffix =".tif", prefix="tcmono")
        os.close(fd)
        url = "file:" + urllib.pathname2url(name)
        C.post_image(url, cp_logo[:,:,0], xml)
        img = C.get_image(url)
        self.assertEqual(img.ndim, 2)
        np.testing.assert_array_almost_equal(img, cp_logo[:,:,0])
        try:
            os.unlink(name)
        except:
            pass
        
    def test_03_02_save_color(self):
        global cp_logo
        xml = make_xml(size_c=cp_logo.shape[2], samples_per_pixel=cp_logo.shape[2])
        fd, name = tempfile.mkstemp(suffix =".tif", prefix="tccolor")
        os.close(fd)
        url = "file:" + urllib.pathname2url(name)
        C.post_image(url, cp_logo, xml)
        img = C.get_image(url)
        self.assertEqual(img.ndim, 3)
        np.testing.assert_array_almost_equal(img, cp_logo)
        try:
            os.unlink(name)
        except:
            pass
        
    def test_03_03_save_channels(self):
        global cp_logo
        xml = make_channel_xml()
        fd, name = tempfile.mkstemp(suffix =".tif", prefix="tccolor")
        os.close(fd)
        url = "file:" + urllib.pathname2url(name)
        for i in range(cp_logo.shape[2]):
            C.post_image(url, cp_logo[:,:,i], xml, index=str(i))
        for i in range(cp_logo.shape[2]):
            img = C.get_image(url, index=i)
            self.assertEqual(img.ndim, 2)
            np.testing.assert_array_almost_equal(img, cp_logo[:,:,i])
        try:
            os.unlink(name)
        except:
            pass
    
    def test_04_01_get_context_id(self):
        self.get_context_id()
        
    def test_04_02_get_modules(self):
        request = IJRQ.RequestType(
            GetModules=IJRQ.GetModulesRequestType(self.get_context_id()))
        response, _ = C.make_imagej_request(request, {})
        response = IJRQ.parseString(response)
        self.assertIsNotNone(response.GetModulesResponse)
        modules = response.GetModulesResponse.Module
        self.assertIn("Invert [IJ2]",
                      [m.Title for m in modules])
        
    def test_04_03_run_module(self):
        modules = [m for m in self.get_modules()
                   if m.Title == "TextDisplayTest [IJ2]"]
        self.assertEqual(len(modules), 1)
        rmr = IJRQ.RunModuleRequestType(
            self.get_context_id(),
            modules[0].ModuleID)
        request = IJRQ.RequestType(RunModule = rmr)
        response, _ = C.make_imagej_request(request, {})
        response = IJRQ.parseString(response)
        self.assertIsInstance(response, IJRQ.ResponseType)
        self.assertIsNotNone(response.RunModuleResponse)
        rmresponse = response.RunModuleResponse
        self.assertIsInstance(rmresponse, IJRQ.RunModuleResponseType)
        self.assertIsNone(rmresponse.Exception)
        self.assertEqual(len(rmresponse.Parameter), 1)
        p = rmresponse.Parameter[0]
        assert isinstance(p, IJRQ.ParameterValueType)
        self.assertEqual(p.Name, "output")
        self.assertTrue(p.StringValue.startswith("Hello "))
        
    def test_04_04_run_image_module(self):
        image_value = ImageValue=IJRQ.ImageDisplayParameterValueType(
                ImageName="MyImageName",
                ImageID = "ImageID",
                Axis = [ "X", "Y", "CHANNEL" ]
        )
        
        r = np.random.RandomState()
        r.seed(0404)
        image = r.uniform(size = (20, 10, 3))

        modules = [m for m in self.get_modules()
                   if m.Title == "Rotate 90 Degrees Left [IJ2]"]
        self.assertEqual(len(modules), 1)
        rmr = IJRQ.RunModuleRequestType(
            self.get_context_id(),
            modules[0].ModuleID)
        rmr.add_Parameter(IJRQ.ParameterValueType(
            Name = "display",
            ImageValue = image_value))
        request = IJRQ.RequestType(RunModule = rmr)
        response, image_dict = C.make_imagej_request(request, 
                                                     { "ImageID": image } )
        response = IJRQ.parseString(response)
        self.assertIsInstance(response, IJRQ.ResponseType)
        self.assertIsNotNone(response.RunModuleResponse)
        rmresponse = response.RunModuleResponse
        self.assertIsInstance(rmresponse, IJRQ.RunModuleResponseType)
        self.assertEqual(len(rmresponse.Parameter), 1)
        p = rmresponse.Parameter[0]
        self.assertEqual(p.Name, "display")
        self.assertIsNotNone(p.ImageValue)
        image_value = p.ImageValue
        self.assertIn(image_value.ImageID, image_dict.keys())
        rotated_image = image_dict[image_value.ImageID]
        # Left side of image is now at bottom
        np.testing.assert_array_equal(
            rotated_image.transpose(1, 0, 2)[::-1, :, :],
            image)
        
    def test_04_05_run_exception(self):
        rmr = IJRQ.RunModuleRequestType(
            self.get_context_id(),
            "this.is.a.bogus.module.id")
        request = IJRQ.RequestType(RunModule = rmr)
        response, _ = C.make_imagej_request(request, {})
        response = IJRQ.parseString(response)
        assert isinstance(response, IJRQ.ResponseType)
        self.assertIsNotNone(response.RunModuleResponse.Exception)
        
def make_xml(size_x=187, size_y=70, size_c=4, size_z=1, size_t=1, 
             pixel_type="uint8", samples_per_pixel=4, big_endian=False):
    big_endian = "true" if big_endian else "false"
    return """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2011-06" 
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2011-06 http://www.openmicroscopy.org/Schemas/OME/2011-06/ome.xsd">
  <Image ID="Image:0" Name="CP_logo.png">
    <AcquiredDate>2011-11-21T10:52:36</AcquiredDate>
    <Pixels DimensionOrder="XYCTZ" 
            ID="Pixels:0" 
            SizeC="%(size_c)d" 
            SizeT="%(size_t)d" 
            SizeX="%(size_x)d" 
            SizeY="%(size_y)d"
            SizeZ="%(size_z)d"
            Type="%(pixel_type)s">
      <Channel ID="Channel:0:0" SamplesPerPixel="%(samples_per_pixel)d">
        <LightPath/>
      </Channel>
      <BinData xmlns="http://www.openmicroscopy.org/Schemas/BinaryFile/2011-06" 
       BigEndian="%(big_endian)s" Length="0"/>
    </Pixels>
  </Image>
  <StructuredAnnotations xmlns="http://www.openmicroscopy.org/Schemas/SA/2011-06"/>
</OME>""" % locals()

def make_channel_xml(size_x=187, size_y=70, size_c=4, size_z=1, size_t=1, 
                     pixel_type="uint8", samples_per_pixel=1, big_endian=False):
    big_endian = "true" if big_endian else "false"
    return "".join([ s % locals() for s in [
        """<?xml version="1.0" encoding="UTF-8"?>
<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2011-06" 
     xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" 
     xsi:schemaLocation="http://www.openmicroscopy.org/Schemas/OME/2011-06 http://www.openmicroscopy.org/Schemas/OME/2011-06/ome.xsd">
  <Image ID="Image:0" Name="CP_logo.png">
    <AcquiredDate>2011-11-21T10:52:36</AcquiredDate>
    <Pixels DimensionOrder="XYCTZ" 
            ID="Pixels:0" 
            SizeC="%(size_c)d" 
            SizeT="%(size_t)d" 
            SizeX="%(size_x)d" 
            SizeY="%(size_y)d"
            SizeZ="%(size_z)d"
            Type="%(pixel_type)s">\n"""] + 
                 ["""<Channel ID="Channel:0:%d" SamplesPerPixel="%d">
        <LightPath/>
      </Channel>\n""" % (c, samples_per_pixel) for c in range(size_c)] + [
"""      <BinData xmlns="http://www.openmicroscopy.org/Schemas/BinaryFile/2011-06" 
       BigEndian="%(big_endian)s" Length="0"/>
    </Pixels>
  </Image>
  <StructuredAnnotations xmlns="http://www.openmicroscopy.org/Schemas/SA/2011-06"/>
</OME>"""]])

cp_logo = np.fromstring(zlib.decompress(base64.b64decode(
    "".join(['eJztXQlYFlXbPvC+gAQioIDKKpqIYggagomiSOKCoYGGJISmfIoampoSmppa5lou',
             'aWVfaeVXtn9RpqV+LWppu2WppWlalra4i/I+/3PPnMFhGBaXxH7ec133xbxzZjlz',
             'zj3Puc9znjMQkaCrAJmSGW2wcTXuaYcdelzlNJDR8WrftLJU0/Vvh53rVyvVdP3b',
             'Yef61Uo1Xf922Ll+tVJN178d1zbXXRwtx9wcrU9ik+FwEafauW7HNcF1Wn60WnBy',
             'sBT3bRB6op7F+X98WgDD4UT6EvFnyhxxpOds8UvyVHGo+2RxsNskcTBxogKZSrl+',
             'IC5PHOyUJ36OHSMOx48Sv8SPEIdjR4kjHYER4ij//q3jcPF7h+HiaOxQ8Wv7bPEb',
             '42jsEP6bIw7z9s8x2eJgu0yxv+1AsS96oPg+aoD4rk262BnZT3zWuo/4vHWq+LBl',
             'D1HUPFG82jzBzvVajjLtvvS3asHT6rL3xx7TaXX0QJubxanYUTg8zKc727lux7WM',
             'Mu2++Jdqwcvqsnlf6gI6P+VLOj7oWerrG3aqntV5C18ijOFo57od1yLKtPvDB6sF',
             '1i6vftl9Ctlm72X8QMVjP6S34obaGrnUPeNhcXqDLxUIXXOwW4Gd63ZcMyjT7gsO',
             'VAt1HC3LPkgYq/Cc5v5INGcvlUzdQWeHF9GqyL42L6c6Z90tTi8KqeXtXLfjWkCZ',
             'dp+zt1pwdbQ++l7nfJXr8/Yz1/cRPfg90cxdVDJpO50Z/CKtap1qa+DkdtbT6ryJ',
             'L90L2sbOdTuuGa4/uKdaULk+mmxsz2neAdW2s56hWbuJpn9LNOUrKhm3mc7mvECf',
             'xQ2jRK+gE6xtjjo6OKyVvLdz3Y6a5frsvdVCXYvT6m1dx6tcX3CgVMcotn3GLqJp',
             'XxNN/oKIbbxtzHt0ju387z1mUIZv2Hk3i9M5L6vzDr7d3Yymh9rni8Md7Vy34+9H',
             'mXaftataqGd1fv3r5Klkm7tP5Tp0jGbbZ4LrO9m2f0lU8AnR+C1EY98nW956WpWQ',
             'Txu6jqO98Xk0JbhtcSNn91OeFucfrA6OC/n20T/H5tm5bsfV4frMndWCt9Vly/7U',
             'Bcz1H1WuK7Z9v2rbNR0z9Svm+qdEE7Yy1z8gyt9IzyZNpPdveYhsbOfPZz5NZ/su',
             'pJ87jaYlTTuej3RvcIK1EfT9J1yU+xg3M9yOdsi1c92OK8/16d9UC95OLnuOZD6l',
             'cnzhwQu2XRujzviOub6DqPAz1jEfE437UHJ9ksJ1GvoqEfOdsleTLXMllfRfTsX8',
             '7pzsPpl2x2TR0tAOJYle/sfdWe+w3T/AWn8NF28yow8jktHAznU7LovrU3ZUC3Wt',
             'zidP5b7OPP/pgl1Xxqj7TDT7NtYxm5nrm1Sup85hrr8muf480aBVRHhvMlYQMedt',
             'aYuppM88OtdzBp1JKqTfOuTSlogUWt4kriStfsiJCDfvE8z/M/weHGPdv5v7gfe4',
             '6DMY0xi3M1J0iGA0lmhg53rtRpl2n/Z1lYCu8HR2O1fMY85Snhu5rmj2b0rHp4pm',
             'H8Nc717AXJ9LNIzfkyEvEeWA688y158u5TqlLyXqt4iI7TylzCFbz5l0vvtUOtet',
             'gM4mjKOT8aPpL+b/LzHZ9G2bdNrcsietaBJHi0NiKNcv7FSqd/CJnl6BJ7p7BhwP',
             'd/U64e/sdpo11ynWR2d4TP0Cl99V6OLVarr+7aghrhd8WSU4Rcd4BR4/N/FjVbco',
             'OFDOz673xVwq120ps5nrM6iEx8HnmevFXcfTmc75dKLjCPojbhj91n4wHWo3iPZF',
             'Z9Du6P70TWQafdE6lT6J6EPbWqXQ5lY9aTv3CfkNw8+z/d/NZY9n+AjV12/nei1D',
             'Wa5/XiWgE8aFdjh1HjyGHcf4VME+6XfcI/X617Rn5Bv0etp82pL5OBXftcGc61kG',
             'rqctYa4/zFyfR7beD5Ktx3QqYR1/LnEi2/WxdLLTKMWu8xiVfmZtz1qd9kSB5/2I',
             '9Tl93Ko3sUanTeHd6f2WyTQjMMpWz+L8G5f7XkYHhofdrtdOlOH61K+qhIuj5ZGX',
             '2mXaSqBR4GMEv+doPP9Bsen7x2+kzKR0mjZhGq1a8B9aWLiY0lMzKLp5JGW3SCqv',
             '1wc+RTTgCbbpy5jri4n6qvqFWL/YdDb9dKd8Os42/ffYoXSYNcwB5vn3zPNv26j2',
             'fBvb8C2sacDx9xgfRvQmD4vzaX60xUKdw/LV23Q712sXyrT75M+qhLe1zicHUtje',
             'zvxO1SuIEwDn8XfWbprY9V9034T7aOcr+2n/U8fp+yWnace80/Th7D/p7gHzaeTt',
             'kyixbSf6HGPU7NXM9ZXM9SeZ649J/cI2ncempNl0Hp8WJ06gMwlj6ER8HmuXofRr',
             '+xz6qV0m/RA9gHa26Sd1i8rzDxSb3oM+bd2H6ltdzvNjPcNIZzRiWOxj09qLslzf',
             'XiVcLNZzp2GTocnhS4dmAednwZ5vosK7C+nAmqN07PHz9Aeb6IPziXY9QLR9ClFB',
             '71U087a1tOzuj2hA8kAay3ZXsekZbNP7s03vt1g3JmWd3n0Kj0knMc8xJh1Ff7JG',
             '13j+fRR4nqbolo9Yt2xhbf5+eDJtZO0CjZ7iFXTe6uC4kx8rl9GUYTXy3M712oUy',
             '7V74SaXgdGOMp/+x4tEbVE0Ovmvg3xldbqUdL/1Ax1ecp5Msvf9YonL9u1lE2yYT',
             'Tey5imb0fZeeHHKE5g/ZTrfdPJjyoGk0nd53Idt0tve9Z7F2mUbnkwroLGuXk51G',
             'qzyPYZ63VXkOff4Z227oc9jy9yTPsT09oI2trsXpsNTo0Yw6w9xjzahe4/VvRw1x',
             'fdL2SuEoHGYsbNGtuARzoYgDwPwoOD99J303+k0qvKuQfn76GJ1kOXKczfTvi4gO',
             '8FB05wyiLffaaEKPlTQ1dT0tz/6V5g7YSfelr6MukT1pXFiSHI/O1WmXyYpGPxUP',
             'nufSb9Ke79F4HtFHsefg9gbm+Dv8zrwbnkQfs013576HH2cRoxujHsMhr1ELO9dr',
             'Ocra9W2VAuuR9rCmsBV+qs6Lwq+IMSpvF3b+F730SBEdWX6Wjj9K9Ceb6cNspvc+',
             'RPTldKL3JtpoXPJKmpzyNi3O/Ilm3fo53dv7XcpLeop83f1of/fJCs+pxwxVo4Pn',
             'nVRfOnyLB9sNoh+Y518zzz9lnm+VmuVd5vhbYYlUxNjKmqiVqxd4/hrjNqFqdMfh',
             'Pk0EYOd67UaZdh+9qVK4WpyKT8NvAn8jYrumfKWCfw+J609rH9lKvy4+T0cXqTzf',
             'z2b6W6lf3hlvo3zm9aReb9L82/bQ9NStNCG5iEZ0eYb6RhdSS4+AC2NRxecymo51',
             'GE5HY1Uf+t5o1Z5jDLpZ+hTXhXWjouZd6bXmCbSWOT8jIMrmbnE6xI8yhgFDbh3h',
             'FiA02Lleu1FWw3xUITh16dagyV/nhr9FBLuOWJfJn6t/Cz6l/u16UdGD2+nAvBKG',
             'as+h0z+fSvT+RKKifBvlJf6b7k5+jR5K534gZRPld3uRhnVeQVkdFtINjRNodHBs',
             'Od8ifOgazzHmfD+8h6JX3mQ7/jrz/JVmneklxoesZ1wcHW1czqUoK8N9uEcjoYed',
             '67UbZdo9f3OFcHO0PrYmMs1WgtgAxLhM+kSN2cW8KP9Oi+pBz0/+kO14icLxr+8n',
             '+ox5/mEB0dtjiV7KO8e8fpxGd1tD9/f9mMazTR/ONj2n41LKiJlNqVGF5OHsRmc6',
             'a77FYfSL5PnXkX0V3zn85potf5Vt+YvM8ReadqK1vC+9QUiJxcEBQQw5Qq79y20U',
             'LPSwc712o0y7526rANsRs/7rn/CTIGYRY1PECEz4WN0ev4Vyom6hR0a9TtvuO0ef',
             'TCH6eDI0usrzV0aW0DNDTzCvF9PwhKepsPdGuitpDd3Z6XG6PXY+3dp2Gt3S5l5q',
             '1TCOJgTdqPAcc0Uaz+FrwdzQuhbdVFveXOX4c03j+W9nepP317M4nRBqLHAUw6XA',
             'o74wws712o1qpps6egX+VZyzRo1FB98Ru4i/4z5Q9s3vnEdj0ufQuvEnadMEovXj',
             'bPRm/nl6me35s8NO0fKsw5QZO5eGdFxG47r/l236SsrusIj63ziTeV5APVqPoYSw',
             'IXRjvQAlrgsxLtAtsOfwJ2o8hy1fzRxfFdpRwTvhSRTu6onx6CuMWxje1X2omk41',
             '3fa1DZz8Gcsrg9XBYdfN9ZvQUB4TDmudwugjkaIiojf1Ce1AAQ1CqGvLQdQ9Yijd',
             '3OpO6tZyMHVpkUPxzbMorultFOrTjq73i6MI/0Rq0TCemvm2p5AG0RTkfQMFeLWk',
             'AO8IsjpaKb1+KPVn9PMOpj5egdTL05+SPRtTkkcj6urRkBI8/BR0499t3eqTk4Pj',
             'SS7nZsZ/GI9X9TzXAPztXK8RrleZ6jg6lZxOW0SUt5Zo1HrGOxLrVeStIxr+JgXW',
             '96fZt33Edvw0rbzzOK3IOUpLbj9I8wbsohl9t9EtUQVsx2fQsE4r2MZDu0yl3jeM',
             '5/diJCW0uJPfi6HUzLslPc72GvNEiOHCOLRI2nPY8aebdKAnm8Qp2MD23tPiXCL5',
             '05XhXq0HukZSTbd9bUOZus99qxzYZs6eHBJz+jx8jblFzOkihdcq5G/ELQ59lQZd',
             '35VS2oyjxZkHmOOHaGHG9zQ7/SualrqVJvVaT8mt8xW9MijuYUpvdz+lRN7DPB/F',
             'HL+T4q/PUhDh34XyG7ZSYls2tOiu+FteapZAzzLPwe8VEtDpYxu2tLk6WvZy0Ycx',
             'ghgOQz0biYpg51vtRpl2H7rBgI3Cw+r06+9JBWTLeUGNwx3yCvP6FXUdHf5i3+A1',
             'Sszinr7zyd8jRPEnzk77SvG3TO7zP7qnx1rF/9I1PJd6ROQrGr1Pm0msde5SbHlH',
             '5niHphkUyzonKqg33cr6Bf5z+Ffgb8EYVOP5E01i+W8svcv5/B7Cx4h12TcxXIe7',
             'NxSVwc712o2yXF9fBpxSB/g2P3YO+gVx5lmr1ThcBbyd9Zy6H7GKMga9l3803dxy',
             'BBX0fpcm9nyb7u7+Oo1MXE1DOz2h6PbE8GFyLJpPXVvkKrY8jnkeE5pGNza5laKD',
             '+9DNnkG0Xo5FX2jWiZ7msQA4/pgEfDAj/FqUODs47hKqjxH61yHLO0hUhiuc6go1',
             'RhjrnJwZXvI3EuLj8XJ5yLzr5G/XK12Iq5gQP4e6Hi9UHwASvmU4nXEPw1N3bB2h',
             'zlnjufH8deXvMvHUVzOV43ruGzoU4Xu83x6KzyMb4sqxlgKxt3pgH+LOByxX4xTT',
             'l9AfKQ+Rk6UODev8BN3V7QXF3zK44zK6PW4BxYb2V7gNn0si2/j45tnKmDWGOd42',
             'OJXahaTSDQHJlOodomgXTaM/ERJLyxW05+04xaYzz6HTZzPaM1yGengLI4Ra52gj',
             '2H18lwDrkpox3K5A9d3KOMrYx/ie8QtjlcwD199nHJR5PzL+YNwo8xFb3ECWwyqB',
             '9wCFxtrYIPn3Whh/oGz3M44zSOJVocYZ7dbt2yguvMvhDKyPwXPj+X9ifCHU9qiR',
             'VI7r2W+VAnFeA3yaHTvXewYR7DrWUDCXlRhzxCQqWKyul1Pithaoced95tA913eh',
             'iEZxCsfvuGkxDWw/V9HnbUNuUfieGP4v6tT8Dh3P+yjapW3wLdTCrz3l+YUp2mW1',
             '1C7guIbnWadn+TQtcVLjdbGWGvbSYbazu9Ag1DXV8MeAa2SCXxlFjExx6bYm2eS6',
             'T+vy3zPJv0HmoZDok44wfmYckmU6LY+DNjsj8z9gZF9iGS83geevifLPMUmoYyTj',
             '/s7yPLyrpw1534oK4qqvRirH9axVgvUINq11LU6//xE/mmw9Z6lxtn3mKuviSoHf',
             'QMpsNWar9yxlHRH1mK7E4yb7hlMM22nMiaa1m6bo88jAHort7tQ8R9UtTdIUfrcJ',
             '6qXkxYSmk+91vrQkuD2tYe2ykrWLZs+XKVyPVfzsro5WrMGArWnHcBnmFSw0cHqA',
             'AX+7vp6PCdUnie9mn9DtxwLaS+V6vCjf1kt1+W8Z8sDd62Ue+vVdJufvkmU8aZKH',
             '74Y4XWJZLzXdK++9jpEm1LUACBiBlskwKaPWb0HL/WHIwwLOi/kfLFc0leN6xhOC',
             '9YjwsDg9t7BJ+7NY44k1cCXM3RJwWMEMCWxPk/lTlHUViMM9n1SorJk71mUcdfIO',
             'pRuDe1Nq1L2KPo/w78b2uxfd1CxT4TV0C+x5ZEAPhevt2eYH1vFU4lswHsVYdJnk',
             'ObCadfpgn+aIBcA38lDXqFOH7LrKuBP1iP9rY6x/7POTxyBBG/xH5s29jOqLNbnX',
             'PF3+fw154G+gzENZPzM5v7/Mh8760SR/9GWU92ITtPZ+ed/7DXnaGORtXdnQj2p2',
             'A1rsqChb9o///iKL5oy7hPpeDhDqOA4xgN2NXC9OWqD4GLt6+v91In4UFSeMo3OJ',
             'E5R1zecSJylrhEqh7FOBNXJYU6GAz8EaaMS1HI8fSV29m1LrRp0Ujd7Svwvr8ZsV',
             'TrcL6UvRQSkKx1uzRo9tOoAC3INoVmAb1inqeBT8XiqB7fUtksiq+l7wPZi2DOeM',
             '+kEC4DRKlOfGy1q+PEZLaJOfZb1carrR5H6zdfmvGPLQt+gdn1tMzs/Q5c80yX/r',
             'Msp7sSlMd19o9YgKjoM2NwZMQ8v/KsqWffPfU8xyCbHciIdCvAg0Ltp4vJHr0OhB',
             'Lh4nD8cOVuLGT/K4FLG1ZzrlK9wtBf9GLCLiyzVgjRxwomOeEqN4rONw5RpApl9L',
             '8nLxopD6UYodh365sUk/xecCDQN/Y1C9ppTsGahol2fleBQch55Zwn9XhcbTmIYt',
             'bdL3ksVomOnZSGSqfnN8B+M3Q91C50Rrx2SW9683l+fpk0Vee7VQ+2r0uy8x8oRq',
             'x/TJjOuzdPkvGPL+NNzvfybn67k+xiT/fZkHXwj6K/hblzCeEKq+aS3LvlVc8JVo',
             'KYYxR6h65Buh9isvM8Yx/HTHXSefd7nh3nsYI2W5wO2hjH8zHhJqf4ZxeW95DXD9',
             'sKia62iUQqH2gajrj+Rz9TY5dohQx0P4f0VYh/OcUMf3QfIcPDPqD+OoUKGOxQpl',
             'mTP1XPe0Or/T3NXzr71tB9LBtoPo1/bZyncpsPYNfEUsuYrcUg4jD2udNfzOxyPe',
             '/AjQXgXWWWCN6I6o/tTa3YfcnK6jZt6tKaJxV2rJuN4nmpyUuIAmSnwuxqNPSZu+',
             'JDhGBW+vC08id0driWxfxfcyzCNAAEJ9f428+I5h0Y6Rx1WWgmR76O3wl7rfXwn1',
             '/dBSVVx/XlTO9fUm5+u5Pssk/yWZ96xhP8bg8DXpNXKqPBb/q3CJ4fgtouy4HTa4',
             'nzwe/kHjeMeIRKGOi4z7s+U1qsN1fJftiC4f79Ih3e+nZdm1NMdwvWJGJ8Ze3b47',
             'hfq+Q6eCI0lCbaebNK5Dn7s5Wk8FuLifYD18Fnp5F3MTseOINwRXf1N4m1MK7Dsc',
             'k6PEaf3cLktZT6HhYCkylfWhBxQMpEdCOtATbJ8fDb2JRvqF01Df5jSxcUTpnNEL',
             'MqbrMbbp4PhihecxzP04KvBvbeMy7pfPo8TsDqkbLABOK0zqfZ2WrzuuogRf2Oe6',
             'c6GTusi8d3X78f5o/srL5fpak/M1Wwy9u9skf5DMX2jYj/dwq2FftDx2tWH/M3I/',
             'fLHGMTD8sugf4hgTTO6P7wjeJJ/jPpP86tp1xKLq3yc8K3w0GM/ofZuLdOcUGK63',
             'T5T3daWKCpLOrkO/tpR1+aSbxfplXYvTqUh377MrmXt7ozOYt1ls7zXeqtz9kfcj',
             '7hZr4/CdFqx5/kEH7MN3inYzdrVJp7nM3Wf4PdK+WYR4dMS7YM5I0y7wMS6V9nyx',
             'xBthidTQyRWaBH0V/B/XZbCd1iDMbeTL+mMyKrfrgwznQstrvuAHDXm5cv/lcv1N',
             'k/PBB/zvnZ9M8vDsmh9jtiHvrG5b41CIUNfbGq+ToyvDNkPet7rn7mlybpbu3Ckm',
             '+Ykyryquv2zIW6XL09sc+C01IzXWcE6xyXYvUUHScR11CJ0G3Qat00qo/+ulwNXR',
             '+rGLo7W4g4dfcVGLROU7W99L7uIbRDsj05TYW6wD3aGgr7L9jQT2fcX7vmzdlx4I',
             'bEcrmM9bwfPwZBnXpc4ZPcf2XhuPgt+LJGDj4Y/hdw9zNXny2R3SPAKFBk4bTOr9',
             'Nf0x8riKkrHuf5J1gLpYZsh7Vp7TzuSeF8P1N0zONwL+0U8Z/zKUd4bJsZtkmQPl',
             '8QjYN9p0IEV3nbdN8uNl3i0meXfqzjXaWSBB5lXG9fqyLvR5eL+bSuww5Gm+qdEm',
             '94PGRP8Ff8tgoX7D2TQZx6YPB7UVS4Lg3lC47yLLhfvj23AzWet829DZrXheUFtF',
             '44DXiEfEWmcA31HEOjkA60K1/fgNOz4tIIoeDYmjTcxz+FTeCOuq+BefZ+0CXzp4',
             'rdnyh4PbKYC9j3Dzhq2Cfxl2wy27rr/QQ5i36SaT48wSnvVrk/MrgtZml8t1M7uO',
             'cd8N8trgrem3hYU51zubHLfT5Ljuuvwik3yNz2Zc179zkyopQ2Vch562mZxbEe6V',
             '5+WZ5KVXUD/lUkVcXx4SIxaFtBNzAiGrSnmPeocfKpM5v8PXybV4dlC0YsO3R6jf',
             'IkJs4mYJxOPqgW9bTPGPZP7GKPNBRXKtKGJbVjXtqNjupdKWazzHt3dfYR3P94Ou',
             'xP+cwVyMY46Hn9CjgnrA3LST4TizhOczcuKovB/aFvYiSwLbafK8y+W6mV6vbtuZ',
             'cf0GwzHwKZnNV1XF9REyz4zrI3XnXirXI03Ow1gDPmPMxd4hLtQ3fD3t5XkjTM7r',
             'Wnk1XUjV4frMwEgx0z9STPWDG1XhBcZmIYwM1hWf+TDnEbuC77Wo35dLVmIU8V0i',
             'xOTimxYA9MokHofOC2qnfN/i1WYJSlzXKqnRl0vtAo4vBPg42Ppsn2YlDsLhc1n3',
             'nv08Q4QRsjzGcRZ8NnGG4ypK6wznwp9h9DEak5lef0CXv8aQ95co69t7x+T826u4',
             'p5bMuG7Wf39oclwPXb4Z1/vIPDOuj9Gde69JfnW4Dh+Jsa1ersYzm3E9sdIzdOli',
             'uV7YMEJM8sMQtpTzwYz+9azOe7t7BRR/KtdXQJ9gDTT8K/h2C94F2PG7G7VizR6l',
             '6BbFnkuePyZ96YskxzXgXPltAIwP2zCs/d1DhBmEuQ9sreEYfYIfVotfMZuHGm5S',
             'ZfAH15PbMSbnzNEd+6ohD/4FvZB63+T8bNOGKp/MuB5lctx9Jsfdpss3jnMwPmgs',
             '88y4fo/uXLOxqcZ1xDweMeRtlXngzkZDHvy7Zh/wgX7W5mLN2ijJ5BzTdKlcL2jY',
             'Ukxo1EorN+xfqMXBcY2Tg2MJfIMftUpRdQrr8dfl91vgUxzjF073B0Qq86IazxGr',
             'q41HYc8XBAFtaRn/XsZ57hYn+IGhHRqlejUUFUGoMXZmYy3YDBTWIoEO6hGhasY3',
             '5KNjXHLIcB7sMOYv/OU54A20TYI8xyweZqGuOo3xMPAphMi8imIEhlSz6Yy+IaC9',
             'yXH+ovxc/YMyD77FA4a8Kbpz00zuoc+fbpKvaQrEbhjHn9t155rFzaE+4C/Huwa/',
             'CMZg0KFa7KTRDwNU6Hcxpsvl+lj+O8IXEl7hEN7lONY121q5eZ2FrwUx6NDk8LNg',
             'jDnarwVNZc3+LOtzLSZ9uZwzeljHc4x9V/PxUe71MSbF/EkC47r+3oGiMgj1vVtp',
             'UicA2tUY95ine3z45343OU/vB4b+1WL1epgc+5jueka7DZ9pmMxD+5nFu+j1cGXJ',
             'ODdE4sI7aEywfXobCw5CFxvnRTFHof+Osdn83IO6/Lkm+ckyL0SUj3PEHIA+zs6s',
             'bzLW90O64836kVsreOZy6UpxPdcvTAzzVaYUEROE93KEs4Pl3LSANrQhPInWNFXn',
             'Q/N8w2hy4xsUnq9QNHqsEgOwWGp0jef4i36A3xvEBmI+HGNSh0EeoaI6EKrNRXwX',
             'NGOJoX7Q1uhDYSfqibIJbfSoUGOe0FbgJ3zX8LcjvuUm3bGoZ/S9eH/AW9jPJ3T5',
             '0MpHZB6OQT+haWqM83/S5R+Q+eOr2XTPyftp10Y5kis5HoYAc+uYs0GdanWBuAr0',
             'P31Mzhkuy4SyoT4whtFz71G570f5LDhWG7ejH0Q9w0+8T6jzsrDbxjhN2Bf4o1AP',
             'qGfU9ymh2hTMI+nH8nNN7pdZyTOXSVea68N8FJ6hf0Z8dhzr+B8G+TQ9h29ewK84',
             'nLle0ChC4TnWFy3VjUfB77lB0QqwHmOEX5jN6uAIXyvmrr2HuQWKi4FQ5/wxlp8p',
             '2wV+cszBwLdi9FnoE/QM2gzaFPOEE4U6dx9qOA66CHa4vywjdJY2b4g6yBbqvE1f',
             'ocbc4VgtKKeu/J0tz02T5dJ825UlXDtd3q+vvD/6p+aVnSTU9xrzQ3jHUSewk3cI',
             'lZdmCZpouCwbygifSDddfi+5r58sA8aOWnwYxuCIuRooy4i5OtSFWfw64hCh3fB/',
             'DlHf+ULtMz0NxyXr7pcuy1ZZO5ZJfwfXh/g0E0O8FFrgHQ5wc7SuY01zBv6ZkX5h',
             'dA+PTzEWXWbkeWA0zWFAx8DP6K3+nwDoajj8nQY0CBaXgv4+wSLHgzWOh7l/vabX',
             'QNpRM+tNryTXc+orfIf+83Z0cHjK2dGxpJC1+n2NI8vZ83k67YJ9iFOvZ3H+S767',
             'MNIOmXWCxaUgo06IyHILEgPq2b8jUNvxd3J9iE8ILqv5J+NdHCxnB/s0o+eYy9Dp',
             'i6TvZaEckwLQN8P9wuBTh75Dn1s32yNAXCqy3P1FjmuA6F/PbtdrO/5urmd5BYkM',
             'b4XzmHstYE3ze0Gj1jbMlRp9L/gLn42PUx1Nv8BnbB3n4SsuBeMZw939RLZroMZ1',
             'rPUfL/UVxhSFwnz9MtYFdTZ9Oa79BJ2LZ3S+iHMwBsR86pVcIwd/OcbDaVUdeCXT',
             'NcJ1JIxRbq9ncfplUqMI29OhHUu5DsC+P6/oFyfMu2CsE5hbP0hcDka4+TDXS+16',
             'sFBjXxDLBx8w/AvG9RpIGOu9YbL/n5AwJsczXsy3EhKEuk6i3P9Ou4wEXwDiKDH+',
             'xri4onVNVzRdQ1zHuBz2sgX4XhAQqfjYNa5jvDq+USv4X+CLhU+jXlaDUFFdcHIr',
             'u0+ZinMzcF37dgO4jrkKM65j3/WGfddVsK0lM27B7+As8+pUcaw+4RyXCvKM352A',
             'z1r/3RmMb76Q96hs3bj+GeAXaqX7Xdl3bCwmZcNvfT+CxvhaXHh34GN6TZanqvdJ',
             'y8c1K1pLru+Ly9y7prk+oCzXMScGDrRydbScWhfeXRmrQsMgVj62ri/0C2JJEG9i',
             'HeXZRFQFofqj0L6Im8HaCrQj7Mh2uQ+xGJhv9hNVcx3thHdNi+NCbDXW06Ht4IvG',
             'nAtiSTfJOge38I/Tdstrt5Hn3SHPwf1/EOr6AHDqNXmc5k81JsRU7ZHnLJD7sIYC',
             '8wVvyP33yf2D5H33yDKi3RvL60OPYG5TixMUsl57yGNxf/i24evHesCV8hxox29k',
             '3Rn9mFgLsVPWjxYHP0Vea4fuXohrOyH/YuEA6h++fMQAYR0I5je8defXldsoA9py',
             'mq4OCmSe5osHtPiZ6fK+qGf4UevWNNf7e5eGOmhcRwLf27hbrMVYi4Ex6ivNEvDd',
             'OhQK7a3MH43yYT5XAqHGY2OuoqtsK9hucB39Z/dcN4XK8NliPQd+gHuVcd1BttmL',
             '8vdu2cawHfvltVB2xJXnyv0hQrVBD8rzYFMx/wRbiTHCXqFqi8XiwtxjpDDXUDgO',
             'dhWcPSSv0UuWA9eCPxq88ZPHYB++LwZ+493xlNuwoYjF/FJeF+861t4ly7IJceF/',
             'G6MON8pz/5Jl8BDl+x+8E9PkNrgK38FP8jyUA/M84DLeEfgW3GV9op5eEBf86ygH',
             '5uLwnmlz/bgf3qN6uvrEsx+W5cR4AnNmzeQ18K5tluVHG2CdyUM1xfXB4Drb9Ox6',
             'TbVblHI9x03Z58R6ZXlHD7/TLzZLUHyNHqpWx3M0ym3YTFQFoa6d+MiwL0rWy0qZ',
             'jziDItmGaIOqNMxY2TZIsG+ansG8vzZf9G+hrlNDn4v57teFyoW1su3BJ83RCRuF',
             'b2zgPUNM2DOybGhbo+1ETBns9xrZzp3lPYtkvrM8D/1PsCwn3i+8T4g9B1fAdU2j',
             'gA+Yo8J4db4s78vyWbQ4QcQRbJTbiF/D+6Gt99Mn8O1beR1cH3Nzi3X5L8o6wXOj',
             'r9M0FOaAntcdh/EQ4u5gs2GDEK8AnaOtg4HdekvWAeI28H730tWBkM+NtlklAds2',
             'vya4nl2/iRjhHSjy+e/YBtqreIHrY/k4gJNLPYvzxnENW9lg210drbBfaFuPfL5e',
             'VRCqpvjCsA/zaZjbRt9oNej1z2VZYBe/Fxf6T326W1zgOtpMm1vEvL+2JhRchb3C',
             'GPptuQ/97P/kNvomPAs4BD0EmwaOIr4JfDPzeYCn4Ky21hXXSpT1sVbug81Hn40H',
             'xXrLYXL/v2VZPOUzaroX86svyzJq3yoS8h7gETQX2mSTLi9MlsO4HkrI6+JeiJmY',
             'LLe1hDKCx+hvPtGVAfX0nO44H3n9b2R5YYsQ53uT/I3+QZsThW3S+jY913EO3ksH',
             '3X2url73ZdvqFSLyeHuSb6Ao8A3gv6VrPUu5Pknul3kBFgeH85k+oYj7gabDOijn',
             'id7BoioItf2g66DdYBOxVstN1vtTQu3HYe9ChMol9P/wg4EziMMAP4xjOPD0dbm9',
             'U1wYt6HetXgjvAuIMx0jj8X4YLU8HnUPm4TYBMQ+an0BfJywuejnoTf0c+5I0AV7',
             '5X7YWrQ/5tgRr6K9Q67yGPQHePcwv44+A1oC3HOWz6V9kwDPDE7DBoIXjWWZMR8N',
             'DZUgVLu/VZZ7rKxTXFsf14iE+BNwDt/ZwLeNoMMQ84LxBOIEoCVhVK7X1QNSirwX',
             '2l5bDAl7rsXKwVdzWG6jT9gn1LZEXRfLbVxjg64ssP2ICbpFXhfvtEd1uX4Vkl6v',
             '65ODM2uZAGd3xP1Az4I3F+PrBelhZ7HmAjFC0Cjg+yK5D/2spifAAa2O0W6wN0a/',
             'ArimrUUD77WPhqGNtfgttE9XceGbFLBz6PcRR48XGFoWY0vE06Df19b73CXLhPci',
             'zuRZwJsi+Ry4Huy2FnuDBB0LDmp+PIx18c0txKVPlccMEGVjLmHXC+U2+rOnZBm0',
             'tRctZLlwbegS8BBxXka9jjgU2JA14sL7jzr4r3x+7Z+Do65Rb5oNcZRlxj21OLME',
             'ccEPCVuu98OnyTqYKp8fz4a+VR+XiqTVFe6P2CfXfwDXkVw8rc74Tif61koXQv9D',
             'EniEZwGPPhJl1/RcrQQbj34BNrZZFcf+v0j/EK7Djmvf6b4WvsV8OQl2DH0qbB30',
             'zRhRM98eR38ErVXtdZn/9HQNcR3+rjaV5IMPVnEVeVHTMRp2XD38H7ZrubU='],
            ))), "u1")
cp_logo.shape=(70, 187, 4)
