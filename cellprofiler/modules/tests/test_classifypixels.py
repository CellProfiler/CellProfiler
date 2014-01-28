'''test_classifypixels - test the ClassifyPixels module

CellProfiler is distributed under the GNU General Public License.
See the accompanying file LICENSE for details.

Copyright (c) 2003-2009 Massachusetts Institute of Technology
Copyright (c) 2009-2014 Broad Institute
All rights reserved.

Please see the AUTHORS file for credits.

Website: http://www.cellprofiler.org
'''

import base64
import numpy as np
import os
from cStringIO import StringIO
import tempfile
import unittest
import zlib

import cellprofiler.pipeline as cpp
import cellprofiler.cpimage as cpi
import cellprofiler.measurements as cpmeas
import cellprofiler.objects as cpo
import cellprofiler.workspace as cpw

try:
    import cellprofiler.modules.classifypixels as C
    #
    # This tests for a version of Vigra that doesn't work with
    # Ilastik.
    #
    import vigra
    vigra.arraytypes._VigraArray # throws on latest version of Vigra
    has_ilastik = True
    
except:
    has_ilastik = False
    
INPUT_IMAGE_NAME = "inputimage"
def get_output_image_name(index):
    return "outputimage%d" % index

if has_ilastik:
    class TestClassifyPixels(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            cls.classifier_fd, cls.classifier_file = tempfile.mkstemp(".h5")
            binary_data = zlib.decompress(base64.b64decode(classifier_data))
            f = os.fdopen(cls.classifier_fd, 'wb')
            f.write(binary_data)
            f.flush()
            f.close()
            
        @classmethod
        def tearDownClass(cls):
            #os.remove(cls.classifier_file)
            pass
        
        def test_01_01_load_v1(self):
            data = """CellProfiler Pipeline: http://www.cellprofiler.org
    Version:1
    SVNRevision:11710
    
    ClassifyPixels:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:1|show_window:True|notes:\x5B\x5D]
        Select the input image:Color
        Name of the output probability map:WhiteColonies
        Class to choose:2
        Input classifier file location:Default Input Folder\x7CNone
        Classfier File:classifier.h5
    """
            pipeline = cpp.Pipeline()
            def callback(caller, event):
                self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            pipeline.add_listener(callback)
            pipeline.load(StringIO(data))
            self.assertEqual(len(pipeline.modules()), 1)
            module = pipeline.modules()[0]
            self.assertTrue(isinstance(module, C.ClassifyPixels))
            self.assertEqual(len(module.probability_maps), 1)
            self.assertEqual(module.h5_directory.dir_choice, 
                             C.DEFAULT_INPUT_FOLDER_NAME)
            self.assertEqual(module.classifier_file_name, "classifier.h5")
            self.assertEqual(module.probability_maps[0].output_image, "WhiteColonies")
            self.assertEqual(module.probability_maps[0].class_sel, 2)
            
        def test_01_02_load_v2(self):
            data = """CellProfiler Pipeline: http://www.cellprofiler.org
    Version:1
    SVNRevision:11710
    
    ClassifyPixels:[module_num:1|svn_version:\'Unknown\'|variable_revision_number:2|show_window:True|notes:\x5B\x5D]
        Select the input image:Color
        Input classifier file location:Default Input Folder\x7CNone
        Classfier File:classifier.h5
        Probability map count:2
        Name of the output probability map:BlueColonies
        Class to choose:1
        Name of the output probability map:WhiteColonies
        Class to choose:2
    """
            pipeline = cpp.Pipeline()
            def callback(caller, event):
                self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            pipeline.add_listener(callback)
            pipeline.load(StringIO(data))
            self.assertEqual(len(pipeline.modules()), 1)
            module = pipeline.modules()[0]
            self.assertTrue(isinstance(module, C.ClassifyPixels))
            self.assertEqual(len(module.probability_maps), 2)
            self.assertEqual(module.h5_directory.dir_choice, 
                             C.DEFAULT_INPUT_FOLDER_NAME)
            self.assertEqual(module.classifier_file_name, "classifier.h5")
            self.assertEqual(module.probability_maps[0].output_image, "BlueColonies")
            self.assertEqual(module.probability_maps[0].class_sel, 1)
            self.assertEqual(module.probability_maps[1].output_image, "WhiteColonies")
            self.assertEqual(module.probability_maps[1].class_sel, 2)
            
        def make_workspace(self, classes, scale=255):
            module = C.ClassifyPixels()
            module.module_num = 1
            module.image_name.value = INPUT_IMAGE_NAME
            path, filename = os.path.split(self.classifier_file)
            module.h5_directory.dir_choice = C.ABSOLUTE_FOLDER_NAME
            module.h5_directory.custom_path = path
            module.classifier_file_name.value = filename
            module.probability_maps[0].output_image.value = get_output_image_name(0)
            module.probability_maps[0].class_sel.value = classes[0]
            for i, class_sel in enumerate(classes):
                module.add_probability_map()
                module.probability_maps[i+1].output_image.value = get_output_image_name(i+1)
                module.probability_maps[i+1].class_sel.value = class_sel
            pipeline = cpp.Pipeline()
            def callback(caller, event):
                self.assertFalse(isinstance(event, cpp.RunExceptionEvent))
            pipeline.add_listener(callback)
            pipeline.add_module(module)
            image_set_list = cpi.ImageSetList()
            image_set = image_set_list.get_image_set(0)
            r = np.random.RandomState()
            r.seed(0)
            pixels = r.uniform(size=(64, 72))
            image_set.add(INPUT_IMAGE_NAME, cpi.Image(pixels, scale=scale))
            workspace = cpw.Workspace(
                pipeline,
                module,
                image_set,
                cpo.ObjectSet(),
                cpmeas.Measurements(),
                image_set_list)
            return workspace, module
        
        def test_02_01_run_one(self):
            workspace, module = self.make_workspace([1])
            module.run(workspace)
            image = workspace.image_set.get_image(get_output_image_name(0))
            pixels = image.pixel_data
            self.assertEqual(pixels.shape[0], 64)
            self.assertEqual(pixels.shape[1], 72)
            
        def test_02_02_run_two(self):
            workspace, module = self.make_workspace([1, 2])
            module.run(workspace)
            for i in range(2):
                image = workspace.image_set.get_image(get_output_image_name(i))
                pixels = image.pixel_data
                self.assertEqual(pixels.shape[0], 64)
                self.assertEqual(pixels.shape[1], 72)
        
        def test_02_03_run_no_scale(self):
            #
            # Handle missing scale (e.g. derived image) gracefully
            #
            workspace, module = self.make_workspace([1, 2], None)
            module.run(workspace)
            for i in range(2):
                image = workspace.image_set.get_image(get_output_image_name(i))
                pixels = image.pixel_data
                self.assertEqual(pixels.shape[0], 64)
                self.assertEqual(pixels.shape[1], 72)

        def test_03_01_prepare_to_create_batch(self):
            module = C.ClassifyPixels()
            module.h5_directory.dir_choice = C.ABSOLUTE_FOLDER_NAME
            module.h5_directory.custom_path = "/foo"
            def fn_alter_path(path, *args, **kwargs):
                self.assertEqual(path, "/foo")
                return "/bar"
            module.prepare_to_create_batch(None, fn_alter_path)
            self.assertEqual(module.h5_directory.custom_path, "/bar")

classifier_data = (
                'eJztfQt8XEX1/+0LlncKLQSodJtCCfwKLFAgKP64ImAUpJEfSOSVDW3KBtp0'
                'TVK6oNDlvQJCeCgBUVcFfwH5afAHGv+grFgkPA2CGP3xWAQkQIXwjqD0n7sz'
                'Z3dm9s5jk52ws3fO55NO771n7+PMmfOdx5lzvtF4+JFbbb7T5o5HoZAz06lx'
                'SNqIKbz1LOoYrkdxOQ2XKVz2TYfz03LXavH52fj+LN9xxx5xhMe9kSF4ThY9'
                'Pn8fS8GixiM+0+SVzfi4HpeD02m+ZStbu7raV7S3dXbljle0tXav6Wzrguug'
                'pxHF5/L0t3EGOo5hvUT6O52rvyOboTKMjwdeRTdoUnwPS2YT6G8/PgY9DmE9'
                '4OmZswFdD78+g7jb6KH+T6l12TP/dczSw6eN62UIH4dnsBzMHYDv+ZlixoAR'
                'r36aNkXHcdy+ZXZgdCtUuvg4shKVUD+WqptYO5DAZbIFlTw9621HxwNnivmy'
                'cXwij3iIkB2Ymdez+k3F7wm9wJqLxHzw/PhGIVse9xq+Pk3EVvHEk3vzFug4'
                'gds3sgMzuHagEQu4AR+Hd0VlGh/37YHKerPFZYlDYAeiuH4bcNm/QPQrwP2Q'
                'Sx+z14F2xXyfwqXjgh2A/vPAEvF7Qj955DAxH+hxzefFfPAiDUvFfJVO05xN'
                '0FgC19usccvq/Xf69Om5MzPGLYT3n9nhQ7HNdZ0lWzhLNs6A3yMDMG0aHCMK'
                '4d/NmNYzG86QtC0+9nh6dnpsXIrJcn/ahIjFl4ZtxPxhXPbsKOYDPU3vJOYD'
                '/evfWcxnBsn693C9UHpqRXbtF4//eRAPeuUNk7ce/9uycM6FHr5XZ/PG/7Yj'
                'fu9NCHkiryX4t8Ql9A3mjP/tOf63gHjO9g6qs0X4mR4/xkp3Hi7DuNyV/t74'
                'jc9vvJc4Dh/75ffx981xaTn4271jT3mBlEv0T23PA5/r/zugGpX7j963cQMl'
                'd/eEdUX14H9/mZ1m65WlsOt/vrjed3BQvQOR9Q71Qdb7XAfVO9Tx5rjcGvNs'
                '6yBdmkfc06v3hQ6qd0zu9vg/ns54OlFPPE9W72dumOESx05m5Q1YDxa5fl/N'
                'yDv55pVzMvR5lq/wngr3i4WcLHlce8gIXFfSE8e5eK2i3rHvw/KB3hf/vpLr'
                'Ha4x9e549b7/+N++BM8i/B/vt590kB2Hd1qMy8JcHZLDtdMZfekCfZnnOr40'
                'UX2RtWdfu5N66KC3S7Fb/bMOfE3RjkC/T3i/+Ld3yDLPF35H7IAPnuU8lz3m'
                '9TvV9NHTOQJ/HE8fPR2cXWBzN8Glnz7u4lD66M7BJcIo/3FZ9BT8xvgCbx65'
                'humHdK+Or165+vSz4Tje2tm6qq0bppXLNo/cuwwdD7QBn+5+Za9R/Ur98khb'
                'eTikPPqsPByqvWwFZ0iqXHn425nRPdHx2CdQyc4/9MENDiHvxsMDmE8YPTQn'
                'a4e2/XICHAm5U9C+Das/7e3bysMh5dFvlDyAVrV3tHTFV7Z3t3SsXt7W0tV+'
                'TltuBWFVd+fZLVCuWNOxDK8r5I671rZ3L4vlzsQ725a3L+tuWdvWfnqsu205'
                'ZvPOj3d12lpWd6xs72hrWdnW2tnR3nE6XHa6WlfFV7a1jN8o1jLOurJ1Wduq'
                'to7uwvXuztbu9hXty8b/Xd3RMt5liq32bj5+uj13o6627pZlrSuXkS/jMNeJ'
                '96bOxzvHe2WduRs7fr/LS8E739bWsmz1Gng1qHdifOJL+vVtwCh9g3m9Rnyc'
                'mSdkz/sDJXcR8yVx2TNfzJeGMizmq3TSr1cZo/RKvzwGrTwcUh5DVh4OKY9h'
                'o+SB7PAsZwAf1ywU8w/jMrabmG8Ul4lFQjanBgswtbuYD9are2VA+zGRzD8n'
                'vDcqYT2pF/+nFx/bdfnqJv12J2ug3Sms69bsJeaHeYfYPmI+MA8JyQQqtMPU'
                'vkK2iif9ejVilF7pl0d8FzhDUlDkAfNxxDwHnteld1AERR7F+pEItH4Uy6PW'
                'MHlw1vs+jY6z2K+Nt97nHEDfr9zrfcXvq7v+wobVn2551Ft5OKQ8IkbJA/U7'
                'N83PO/buJ+aH9t23v5gvjcvRT4r5YlAeIGTL++8nDhTzwX7H1EFiPhhnZg8W'
                '86mSfr1qMEqv5PKoyR17X438pcL4d7L+lYvlsCX1vGI5pJxKpNZl3WtaV7as'
                'wus7+WO8nIT2VRILKugYrR51jZ9ZtnrlmlUdBYb2LnppKd7Ztqy9q7ByE+9c'
                'fdrKtlUt3WfHc0s2navXkss1zpou9MOVrae1rewq6N2AYxbJ9jX3uKiE1mPn'
                'm4JFyvYoJLZHeNqSXP/A4z26xVS6PYL9qtCvH/6MmB/aTZLnYRtQKjfuw7ip'
                'oF/JbdEZl7JMlYv7/uPFms+i48gRqJTZ64EjUWntdTCJN5/QdATNV37/YV04'
                'kcHbfszGifxAkUPQXvuOELIFjvTjRALc9I3GiQa8r7X5KFRK4xUdjUqLE8Ek'
                'Hk4kjqL5KhUnwA2c8Fuci86Axwgi03Ci/hgxP7TXoaOEbIGjcuMEhB0g/NLx'
                'VsK00TgRxXENkl9CpQwnQv+FS3xscSJYxMOJ9JdovkrFiTC+X6EdN+ImkaSe'
                'ZxpONB0n5of2OvYlIVvgqNw4AdsNCvoVPR6dyVLPrVScQJTbNzcy//1pLnEc'
                'vXnpzAxxPHbBiduS13n38bnOHu/pKvy+hLgGEH9Cvu/bqz1eHAJcpRCHwIU2'
                '5F3zFs/mEr/zfuPV/U7E73DcEWrft//35MrszlGQL8RB4n2fUtwAdp9/+qmx'
                '9WWK88CP/1Eu+Xr78ecSPHhfvbst807MNxIxWeol38DqEMvnH0OFfz+oM7lM'
                'PN3hxbrBpBzrBviZWDc8nWNj1hCxH9jYJb7fy8YacfcYe7lMMWVksR/qXP/z'
                '/JgypHwhlgNQCbEcWJ1jZcjqYAl2kY2Tw/JDnfDuw7cDrEy8bRVlkElOT72t'
                'HGR4K0/nFoz/EVvjQGYQb8XF/G5hm50vpiTXv//KBGVZ5/qfp46HntuU0l8i'
                '9ojMJsr0F+wJ7/esTS/ch60rLzQrYTNdCBnt1YUnv60JXkFsnBIxCZFs/Oe2'
                'Fp7rkR3/BZNaVse9fc35qK0tbYnultwgDx0f5+1njuByX1zuh8v9cbkElwfg'
                '8kBcHoTLBlweXLBbDT7vIiJ940b3y+hMknqeaePG1GlifmjnkajGlzKQ9I8b'
                'M1i/stRzK3XcKMWN01FpcSOYxJs3HG6j+Sp13hDCrRFxek5AZ5ZTzzPO/seE'
                '7AX7v0LnW5lH5bb/EJeyoF8zcVOByPuITLX/qQ5UWvsfTOLa/zNpvnLbf2Tv'
                'Qnl3q0FJQi7IbxSTxPmG7+jZTMwXxeWAhA/McPYsMR+kywidK+aDzxz6hJgv'
                'b9+3EPPpG0fdiUWZpJ5nGo4OrRbzg5wTq3S+lXmkfxy15CvoTJZ6rrE42o1K'
                'i6PBJB6Ohpn8SeaMo+aciM6YPY4aWiPmz9v/LiFb4Ej/OCqB9cuMcRTPTy+0'
                'Fh3X4xYtw4n+c1BpcSKYxMOJxrNpvkrFCQjbR8THOAmduYl6nmk4MfY1MT+0'
                '1/TZQrbAUblxog7ft6Bf67F+1VAW0TSciOB5gaZ1qJThxHASlRYngkk8nIiv'
                'o/kqFSfC+H6FdvzkyehMknqeaTgRvkDMD+11cJ2QLXCkfz4pciE6k6WeW6k4'
                'gQj51vWHt3DRcZ0r4HM4vnKjt/W8qui3iPIh1vzoPkXfOpk/N9+3jrv/FeeB'
                'br4ElbI4yj3foF7UiY+gko4SZalaiZfnPPMEKnl6NvxndDz2FwnfS5jvZfq5'
                'PP4xrLe1WC9l+blHLkcl+Mdlvo/KNHzfLai0/bjqJF5+7qbvin5V/vzczfQ0'
                'WxFB/zD1GzFfPj/yejEfvMjwQ2K+SqcpiKf5TThDUqX2W9g47sOXifnDuIx8'
                'S8wHeup+W8wH+td0vZjPDJp8fm7YswQk27MEexIgl6/fngS4JtmzlO08ntpf'
                'Q/Rj+XmmydK98Wq8X4fdL+Nv7zLnXn+v/3X/+/PvV549YfhcyfLFVOqesJLz'
                'mCevW82MC4RyI8YVsvvLxg3q+59Ee2o8+XhyAzn77fME/gnIl8j/zN8DRJSE'
                '/NXyd2fOXUvW38Dm54wp6med638eiJ+/mae/ICfVPY2g7+SeRtBpzp7GHI+3'
                'n21h4fe8/WWOc+o6UjbRl954hdHV4m8jjpsO/pDaj+a4J6wr035H/h5a4nzR'
                'XkP+/dT3l4lyXQORewGhPlRyXUN9KO55nmxudJ/zQGr1e8bxD+Dn8W217HkT'
                'sTUgU789z/A7cv8pb5wY/gdmesP/racgnrZR/Ur98nCtPBwqvoqVh0O1l2/A'
                'GZIqVx7+dqfnx+i49wZU8vJzp39J3o1nQ03Kz+0aVn/a27eVh0PKo8koeQDZ'
                '/Nw+143Iz91slL6x+bmjvUL2/P6Z8I1iviQuI98R86Vx6d4k4qp80q9XUaP0'
                'Sr88YlYeDpWv0srDofzMjZIHm5+7/3tifsjPHfqBmA+iLdf+UMwHXpb1PxLz'
                'wXp1w81ivo+LZH4lmVtRCetJDXeiEmDPrstXN+m3O0kD7U5hXbe/T8ien3cI'
                '/UTMB/3w2tvFfNAO6/9HzFfppF+vUkbplX551NwIZ0gKijyK80c24Xldm58b'
                'yaM20PpRLI+BG+AMSZUrD/95budXqEhivzZu/HvG7838/NwZw+pPtzwGrTwc'
                'Uh5DRsmDzc/d8FMxP7Tvxp+J+dK47PmFmA/i+4TuEPOB33ntz8V8sHup/n/F'
                'fDDOTN4l5lMl/Xo1bJRe6cvPncVysPm5PQabnxuRbD9v5B5UQuux803BIn37'
                'amN4vJeknlfp9ojdV5v4tZgf2k34bp1vZR7p31c7gvUrSz23cnHff7zYj/c1'
                'DWVQKbPXzfeh0trrYBJvPmE0Q/OZE39h8Ep0Jkk9zzScSP9OzA/ttfG3Ot/K'
                'PNKPE41XoTNZ6rmm4cQw3tc69ntUynAiOYhKixPBJB5O1D5A85mDE/U96EyS'
                'ep5pODH4oJgf2mv8ASFb4Eg/TvRh/cpSzzUNJ5yHURF+FJUynOh7DJUWJ4JJ'
                'PJxwH6X5KhUniuN+pq5GZ8yO+zn6BzF/Pv7io0K2wJH+uJ+7XoPOmB33s/5x'
                'dNyI41VJxxN/QqXFiWASDydiT9B8uvxYbP7QJdeiM0nqeabh2uBTYv78+OdJ'
                'nW9lHukf/9yJ9StLPbdScU2KV/+HSotXwSTu/Ndfab5KHdcU57257jp0xuy8'
                'N4NPi/nz9v+vQrbAkf68N8PPoDNm5L1BlIsHMjL//WkuOl7kCvjyZfi8S0uJ'
                'V0fEn3Mk/HWu//nimE/jXdKi+H1efDI2ppYXp2w2wefFhPLa0vYFNhe3GYh/'
                'lqvpBeN/8wu/y8WE8kr4nXff8W5yzi7Cvb0+wT7jf3s5hXhTOD6ai8fB7iJc'
                '7pl/vH/csWNPoWIwRmff+hR5TMT1UouxKIuz9fxpN+Lns3XE3ocfo444Hr1v'
                '44aPKY4bXC++b7libMK1ScSAZOs3fuaGGa7s/YmSqH+2zfrKr3/Wga+VSf78'
                'mJO8uHdkGxXFvcPEi3sHbdDhtBk2fv7YBSduy31X+vxi1/88EC9eM1Cd63+e'
                'r3OkTDzd4dktMpadxG6VGneUlRcRV5TVscnGDWVjJbJ8cH/e79nnc9/DkiLJ'
                'xn+Dr6LSjv+CSdzx3wjNpzt/uPO4mB/yhycuEfPBd6QlfFFcjl0n5oP9ReG3'
                'xXyQP7zxn2I+sNXxG8R80B6HJN9RrnH0Afnn5tcHn0Vn6AlG08bRNRvE/CDn'
                'gVd0vpV5VO5x9N74vgX9Ovk5dCZuxPqgFEdxgB6Lo8EkHo5GmfjRlTqPGsb3'
                'K7TPR7LoTJJ6nnH2/y0xf97+c+J8B5X0r6Md/Tw6k6WeW6n2n+cf0oj7hbF3'
                'USnDidH3UGlxIpjEw4nUuzRfpeJE8Xrb4r+hM2avtzWMifmhvWbfFbIFjvSv'
                't/Vg/TJjvY2HE3E8L9DzISplOFH7b1RanAgm8XCi/0Oar1JxIozvR+R3eAGd'
                'SVLPMw0noh+J+aG9hv6l863MI/3jiQzWryz1XNNwohcjwAD+UFkc5NGZiM/F'
                'x0PL0DEd5clStRIvv3Z4KdIDbh73/0LXm4+X8LVivmV0R4PH34yas5OYSeov'
                'P79246aID/zHwzuj4zQ+7gujY9vPqU7i5dfurxVVePnzaw/8h1jBoP80crCY'
                'D/S4xhXzwYs0HGm2YuuPW9gT8kO0SsV1Ng57wybi+g3jsqdGzJePhztbTf/6'
                'tzVbrxBNPr+2p1083z/w+cS+f+BnReUMBR6Obxqbd5XwvZL56yn5/zmPhf7B'
                '8eObuE+tSCbgz+onE8jZDJql4A9J5RQGHl5OYX7OWvg2llRz1vrKMhJywC+O'
                'rYsJ+dZGn//264r+z+q+lX6+q35+hJycwlT+Z3g3lfzPcB+oW+89Fo//FfyZ'
                'HTeMy13zp3zbBOF3ztarUp756EEdqjgv8/9UrOeLwT8SfK6F9y3yr+Tfn+9f'
                'yta7Vw9l8Kml2ijUKa53x6t3r64XFJ4Dfu+8Nhq9eenMDHGcat3w9gR9cbXU'
                'ReyAD55V9F+X+fry82RP1NeXU1eltFHW1zd5wP7vinx9ibZYx3zPZP3HayT3'
                'k/nvi/OQ6+9X9hrVr9Qvj7SVh0PFObLycKj2MtMsefjPT40uQF8xNgeVvPza'
                'zv7k+Kka8munDas/7e3bysMh5dFvlDyAbH5tn+tG5NceMErf2Pzame3E82tR'
                'XCbnivmSuOzZXsyXhnIHs+f1piC/llF6NQX5taw8HCq/lpWHQ8WzMEoebH7t'
                'mp3E9hDya8fmifkgv3biE2I+WKZJ7SLmg/Xq3vmVaa9lfiXh3dB7w3pS7z7o'
                'GPwP7bp8dZN+u5M10O4U1nVrdhUrPsw7xBZJ7AQuE7uL+aAdpgxvcPr1asQo'
                'vdIvj/jcIMujOP9jP57Xtfm1kTwSgdaPYnnUGiYP/3nu6BL0FVns18bbT+As'
                'pvHE/PzaYcPqT7c86q08HFIeEaPkwebX7t1D3P+D9t23p5gvjcvRfcV8EP8m'
                'tljMB37nib3EfLC7J7W3mA/GmdlIefq7+vWqwSi90pdf28VysPm1PQabXxuR'
                'bL9rz0F067HzTcEiffEJBvF4z+z4BMMNYsWHdpM80DYQkvTHJ1i8GbpkdnyC'
                'mk+ir4gcgkqZvR74tLXXQSbefELTIXrnE/Tlw3sat2Oz8+E5h6rhRN8htmGS'
                'pD8fXmJzdMnsfHgNLnr95sPUcCL7WYsTQSYeTiQOMwMniuPWprZAl+gRuGk4'
                'UX+EGk4MHWYbJknlxgloD8T82Zbokms0TkRxXINkoxpOhL5gcSLIxMOJdKMZ'
                'OFE8nkhuVQ3jiaaj1HBirNE2TJL0jyfmbV0N44nU0ej1+45RHE80WZwIMvFw'
                'YvAYvTgBZPNrL94G9+yo55mGa/XHKo5/llpDQpL+OJ63Y/3KUs+tVFyT4tWX'
                'LV4FmbjzX8ebOq6pb66GcU19VNH+H28bJkm8OFWQv5SXn1kl9hGWtG9+Zohd'
                'xeRn5saSSy6+ioyLFCvEeptQXuXR23peVYwrhWOc7bP2XvhWfz7x7wmS/L7e'
                '9T9fUXGq2Jy02c7jqbzIfXuc8F4pccFKiFMli8UHcRonHwcM8iADkTHbQOak'
                'fEHfOXmQHU/XvTqDPpLDxOrz4vEtJO7t1Y0Xn6+euDf+rRvGJROrL1H3JhWb'
                'j62n5EcnvcboMUuqMRjrlH5fyJetFh+OjY9598K3yqQX/PhlpcTThGeVKZ4m'
                '266K4o+q5xdXyyfPj5k5WbvFb3eV0q6AyHYFOMRvV9VN+uc11zebNK8pG//V'
                'r7DjvyATd/zH5FPQnV87erRYwSC/9vA0MR98hzNdzBfFZTIm5oP9RX0dYj7I'
                'rz3ULeYDfByaoza+i0i+Q9886pNfqYZ51PjpanKubbMGjiT986jHnWjSPCpv'
                'fbAP24/BM9TWB5tWWrwNMvHwduQMM+Zbi/0Nl5yELpntb9i7Sg0n3DNtwyRJ'
                'v7/hBqxfZvsbDuH+42hcDScSnRYngkw8nKj5qhk4Ecb3I/wyTqmG8USmSw0n'
                'Yl+1DZOkKfDLOKUaxhNjeP6gdq0aTqQTFieCTDycaFhrBk4UjyeuOLUaxhMj'
                'Z6vhRM9a2zBJ0j+eGDvHpPEEIt/8gxrXwiEfrfD3hI+E0lptUR665HWrmfeZ'
                'qA/KHJf3e+7+r6/j3uh5JM7y400PnI+1Bh9nsjTuWqpu4uUxH3kY6UEZ8rq6'
                'dEQyRDz9Da/D+ng+qb/8POY1FyE+8NMfuR4dp/Fx/Lu2n1jNxMtjnviWqMLB'
                'npYvj3myT6xg0P/s/4WYD/R4+B4xH7yI8zuzFVt/fMjmi/0QrVL7RWy8e+dC'
                'cf2Gcdn8TTEf6GnsSjX9S1xltl4h4vW71POYez5pKr67nq8akQfa8XzgPGzb'
                'vvBbFg+pfMCen9sC4p7e77w6W1Q4BT6IMt9Sd+FXNskQx6yvouPcspY8Tl7x'
                '4uuMPaTuRxB7neWrk/zet7/s7jH2MvC5/r/Lk+T+kBtXNXdu4T68egcS+Jbm'
                '2qrnt7idU6h/Vf96uMb415fo+8j68KJ67Q9vgb8XxhHF302W7o1Xk+OuR0IX'
                'QH3Lcg7njtNPja1XHPco+vKfe/29/tf9359/P/DVl/u8ito7nJtk3ndpe4d7'
                'KrZ3Nhf16H0bN1ByCZ93KT5WqkfC11jW3pTGt9Hnv/06zUfbH97vHLmeQB57'
                '3n34ee5V8r6L2rtXJ9sR/Kr1DnnfoY4l+2liB7z8Dlm3I7/ffRv8TTJbLPMj'
                'Z20Cz9aL6yh69oX4/fhtzP+8OrbwxorOMJJ++K/FfRX9/cqoUf1K/fKIWXk4'
                'VP4fKw+Hai/nmyUPf5sz8D30FZmrUcnLYz74M9ImVUMe85hh9ae9fVt5OFSc'
                'VaPkAWTzmPtcNyKPedIofWPzmKd6xPNrUVw2XiPmS+Ky+VoxXxqXsevMntfT'
                'r1cpo/RKvzx6rDwcUh69Vh4OKY+0UfJg85gPf1tsDyGPeeQGMR/kMXdvFPPB'
                '7uqm74j5YL06elNl2muZX8nID9B7w3pS9HZ0DP6bdl2+ukm/3ekz0O4U1nWH'
                '02LFh3mHyI8kdgKX7s1iPmiHTbeY3eD061W/UXo1Bfk1rwmyPIrzbCbwvK7N'
                'Y47k4QZaP4rlkb3aLHn4z3PX34G+og/7tXHjPzN+b+bnMR8xrP50y2PUysOh'
                '9lkYJQ82j3n0x+L+H7Tv+H+L+dK4HPipmA/iDEVuFfOB37l7m5gPdkc1/UTM'
                'l98n+D/l6e/q1yunynB0onnMQ1gONo+5x2DzmCOS7RduvpNuPXa+KVikb99u'
                'Dx7vmb1vN32XWPGh3TT+r20gJOnft9t4CbrkUoKvXNz3Hy8O431NYwOolNnr'
                '5P+z9jrIxJtPqP2V3vkEjflpL60GnBi8Ww0n4r+yDZOkKchPm6oGnHB+jV4/'
                'fK8aTvRlLE4EmXg44d5rKk4kv1ENODH6WzWc6L3XNkyS9ONEw2XVgBP1OK5B'
                '4/1qODH0e4sTQSYeTsTuNwMnivP9JS5Hl8zO91c7qIYTmfttwyRJf76fOVeg'
                'S2bk++HhRNOD6PXjDyuOJx61OBFk4uFEz8N6cQIoaHnMi3Ft9LFqwLXRpxTH'
                'P49YQ0KSStwNUY5BTGyOwVysDi+2BpHDkYqzA79TzGPLxiAVxNFgY4xMKEZK'
                '+qh1VIyUsQtO3Bbu6/r+TDlGCj+fJiJ+nlWyZGILEfKRff8i5jrLx4/hwsuj'
                'CzSZPLpYt3h5dHM8nm4udAqxWji6ktp12XuUbAq5QNk4TPz4UyT/m1fOydDn'
                'WT7eMXteFuOFjeGyt+vPJ39+KTmPIUbWJOMoqeY8djLnrqWOC/GQlOIb9S15'
                'bzPXn5/9HbQl1bbG8tVx78/KdzOHagsuGXfOk8/WxDXAIk9+nk579hX4ZXYT'
                '2omi3RyZ/z7kNZa1e9R2Hjro7Qnqqm8suaLYV/z7yeIbyeymelsoNY8uJqU8'
                'utBu/OwW1F1xXX18pH9ce8UfTBrXSser/2fHq0Em7voXE/OsUuc1MW6Q+8Vx'
                'ZHrwSAc+w8Z/TyuO/3xi0wWZym3/IbZiQb9uGkKXktRzK9X+c9e/nsWtJKs2'
                'r1n7gsWJIBN3/SurFyfYfOf1D4oVDPKd7yqJV57Pd/6smC+Ky/hLYj7Yh9T7'
                'mpgP8p1n3hTz5fOrXq2GA2Pnifn0zbfOe7wa5lujL6rJOfQ3a+BI0j/eGnjc'
                'pPEWD297sf0YeFkNb91XLN4GmXh4O/yyGeOyML5foR0P/lGt31zZOJF6VQ0n'
                'IiO2YZKkP49t4xPoUpZ6rmk4kcH9x+w/1HAi9obFiSATDyec183AieLxRP2T'
                '1TCe6B9Vw4nm123DJEn/eGL9k9UwnhjB8wehd9RwouddixNBJh5O1L9jKk48'
                '+adqwInh99RwIvmObZgk6ceJkfdNwglEvv5hrL8S4fs32bx3SvnOHediyFcu'
                'y6eG3ve9yIOKvof8fIzc/M//RNXpfkjiJj/OdOIjzI+Pm49HHjF0FC1L1Uq8'
                '/OV9jUgPuOP4o9H17DESvmMx33HgMecI+bNYb52PSP3l5y8fwh5c4J/fNwcd'
                'p/FxZCd0bPt91Um8/OUN207n/cQp9v1WxQF+/nJ3N9HzCv3J+P5iPtDj3oPF'
                'fPAimUPFfJVO+uNC1k73Q7RK7eewce4zG8WGK4zL2s3FepAfD22hpn8NW5qt'
                'V4hUfZMLpWbfY/ADzz+T2RcSvXnpzAxxXEJfts71Pw/Ez+VL/16+j8RrcYq5'
                'fsF33i1c8825TOQ4lvmxg188Tx58v/mPOzc9PIfMVQ2/n2CualZ/BLnpVXNR'
                'swTnlXJfs/u8nLsXvsU8n/cc2fuVL1c17P0A2ZP16vn4b+2QOFLIVQ1E7q2B'
                '/QOKe2sI+bBjS195svVN7PcoNRf4ZOULNrP4/jz7AETaB5A5Yx9y1zzZzy3w'
                '5H7jyZ7YWyOTb+0hI5RNJ+YGZHZFaR8fuz+G2MeouteMJ989Xd71UvYueXjF'
                '6q8nU0J/ZZhE7Lerc/3flfqWzCUvhzLEcawgE8hpzv4OiD+/Ae8qfj6f9Pcr'
                'w0b1K/XLo97KwyHlEbHycKj28hG9QoeocuXhPz+V2BnVanJrVPLyl6cWk+On'
                'ashfXm9Y/Wlv31YeDhU3zyh5ANn85T7Xjchf7hqlb2z+8satxPNrUVyGthHz'
                'JXFZWyPmS+OyfrbZ83r69arRKL3SL48mKw+HlEezlYdDyiNqlDzY/OW924nt'
                'IeQvH50r5oP85c4OYj7wtqmplazXYL7wjpVpr2V+JX270Ovy4T3QMfhj2nX5'
                '6ib9didmoN0prOv2fkLcrvNxMuarres6C9TWdWvqKtOeqJJ+vYobpVdTkK91'
                '6yDPYxbn12zA87o2fzmSh7NNkPWjWB5pw9oLZ//NXugrYrvR89zs/ooU4/dm'
                'fv7yPsPqT7c8+q08HCpeh1HyYPOXhxeK+3/QviO7qs0nJv5DzAdxg0Yl/rHg'
                'd+7sLuaD3U419WI+GGfG9ixPf1e/XmWM0it9+csHsRxs/nKPweYvRySN37cv'
                '3XrsfFOwqFz7cHfE9yPWP/B4bzn1vEq3R0Xx3/YT42A+/lvE7PmhclO5cR/H'
                'fyf0606sX7ADD1Hl4j4n/hve1zRwACql8d8OsvY6yMSN/3aA3vkEfXG5l8/A'
                'MyXU80zDiVSDGk5EDrQ4QZL+uNy3Y/1KUs81DScyeF9r9lNqOBH7tMWJIBM3'
                '/tshZuBEcf7yxTNxT4l6nmk40f+fajjRfIjFCZL05y9/EeuXS1lE03BiBMc1'
                'CB2mhhM9n7U4EWTixn87zAycCOP7Fdrxrpuo9fcqGyeGD1fDieRhFidI0h9P'
                '+masX1nquabhRM2R6CsijYrjiS9YnAgy8XCiqXFq/FiClr88jO9XsDsXbVoN'
                'uNZ/lOL45/MW10jSj2szQybhmhSvmixeBZm4819LzRjXFK+nvxiqhvX0/i8p'
                '2v+l1v6TpH89vXEzk9bTZfa//wRr/4NMXPt/vF77z+YlHTlCbMcgL2n0Q7Ei'
                'wnekJHxRXIZOU/MvjsTEfJCXtLlDbd9d89Zq9n1A8h36xlFjzdUwjhqLqsk5'
                '/WWLoySxMQG92Im8mItYchBzMSdTiGkJ2kvGtMTnVGNaplo3MDEpb71QMb+B'
                'b8zb5EcnvYZjNjrM79n7yGIuymI6Lnb9f1cs392c4lixC53iWLGcOMAi+eZa'
                '+oLxv/nE77B/kYv7Ny6OgenuQn8DGx80xsaj/O6O+bitruNLvPiWbLxPf1kX'
                '4s/K4sPC83mxG+dIrodd//NqcX29GLxEW8jVlYelswk+r668mLzbOwXy2ooX'
                'g3dn4rdeXdY5yCYrthVZXF934Vc2yRDHTubctWS99sz5/INkPRXFHU1et5o8'
                '7t1wDht3l3oeSS7nPCZZ3FReDH0gWYxP2f3V485CrG6y3iFWN1nvE4zVzdop'
                'to6iL73ximJbALskkxnv92xcVR6fpWol/fOoia+YNI/KWx+M4PFD03K19cHh'
                'NjveDjLxxtvx5WbMtxbnEWw+Eb252XkEw6erjRMHl9txIkn68wi+g/XLjDyC'
                'PJxoxvNHiTPUcMJZaXEiyMTDid4zzMCJYr/0mSdXg1964yo1nBg9w+IESfr9'
                '0u85uRr80pN4/SAdV8OJ+k6LE0EmHk5k4mbgRLH/xr9OqQb/jXiXGk7UftXi'
                'BEn6/Teip5rkv8HDib5u9BWDZ5E4wY+X7JyD0REfxoZp3LBU3cTLwx1Zj/SA'
                'p2eNQ+h6+GIx39Cf0PXRP9P2jMcfW4v4UmeT+svPw938dcQHfuaRHtxPwscD'
                '37L9nGomXh7uzDdF+Fn+PNyDaTFeQ/9p7Kdivnzc+rvEfPAijXeb3U+Ygri4'
                '5/ohWqXiOhuvvfFr4voN4zJ9sZgP9LT/EjX9y1xqtl4hmnweblHOU8jZC+TV'
                'GSfnKZVTFnw3GJ+NHBZ6PiQLiXt6/J4PST3xO1mu5kIeVb7vClFmO4+ncjkT'
                'uX7ZnNa+djL91Nj6Ijn6P1fmE5I7Dr809Bx5n6aDP3yFc99SfUr411Xyr4v8'
                '2qDeoY4Yn45c3CavbmG92yOv3r26XkC8H4xLyHqHa5J6T9S9OY36vkzXDdhP'
                'hM35TP0OyuSbV87J0OdZvsK7KNyP9Qkj9Eopp3fPmvnwe9V65d2Pbe8FUvG3'
                'g9zspC+PKDc7vBO24b652Tn1zvO3K/LTci5eTbeJ+DryeOT3u2/D/WbmvMs5'
                'j0nmM5Q7P/TcplBXdZL7yeqydL8rILKNwrP82ihc82ujXv3Pcwr1StYVtG1J'
                'XbHtkLDHSu0welCHar9UzU+V8RfLRs76t2LdA36U7i/Gm390mX6Krv3OUxCv'
                '36h+5RTE67fycKh4/VYeDtVezjZLHv7zU871+CsuQyUvD3dNH2nnqiEPd79h'
                '9ae9fVt5OFQeCaPkAWTzcPtcNyIP96BR+sbm4R5KiefXorjsuUzMl8Rl+nIx'
                'XxqX/VeYPa+nX6+GjNIr/fIYtvJwSHlkrTwcUh4jRsmDzcMdvkpsDyEPd+Jq'
                'MR/k4U5dI+YDb/XeayXrNTCeuK4y7bXMryRyI70u33cz7X9o1+Wrm/TbnVED'
                '7U5hXTd8g7hdw7xD4jtq67qpm9TWdXu/W5n2RJWmIO+0UXqlXx7Jy4Isj+I8'
                'kRk8r2vzcCN5pAKtH8XyqDdMHv7z3PFbsd9mmp7nZtfzan4wNet5hffVXX8R'
                'w+pPtzwarDwcUh6uUfJg83D3fU/c/4P2PfB9tflE57/FfBAnL/EDMR/4nad+'
                'KOaD3T29PxLzwThz9Jby9Hf161WjUXqlLw93E5aDzcPtMdg83Ihk+13Tt9Ot'
                'x843BYv0xScYxuM9s+MTjPyPGAeh3fT8xOz5oXKT/vgEPefhHZiUZapc3Pcf'
                'L4Z/hr/iDnLfHt9eD/7c2usgE28+IXqH3vkEfTgxtK4acKLmTjWcGLjD4gRJ'
                '+nEilawGnGjE+1pjv1TDidEBixNBJh5OpH5pBk4Ux8UcPB+9udlxMRv+nxpO'
                'ZH9pcYIk/XExl1+AJG52XMw4jmvQ82s1nKi91+JEkImHE/2/NhUnui+sBpyI'
                'ZtRwIvQbixMk6ceJmouqASd6f4tH3b9THE/cb3EiyMTDieHfTY0fS9DycBfP'
                'kzU8UA3zZA1/UBz/rLe4RtJk88epxFeCa6XGV4LnSeLs1B4ywsQ5yuedk+WU'
                'QvFAluwO8UD4sU6IkogH4qjcX3Y/IlYLG0tGljNL9nx+rBi23r26YePscGK3'
                'uNCWvGuevswl7qNQ72zeQOex0D/I7ydymoUl3yeLb8XGOUJ8106fge/Lxkth'
                '7yvLWwjxuNTjV5HyhZxkQJPISRa/8fmNZEwZIh9f7njsghO35b4rfZ6NicPy'
                '10nuw8+1qCIT0Dl8ztfWgF2QyITNidj88I1jpdgFIk6PLOaSTE/4edp49ldV'
                'JpC/E4jMLwlyUszfWaRDhXapJi9G5xzn3LVM+5yo/EAnJ58TcqLxA0H2fvgG'
                'cubLt3JJ/zrZ6AMmrZPJxquJJ+x4NcikL4935EGkOUnqeaaNwzJPqo3DYn+0'
                '4zCS9Ofx7Mf6laWea6odzvzV2uEgE3f/27DeeUNk70L5/Tvx34rtWBMuR85S'
                '2+dTs1bMF8Vl7TNq+3zcF8R8ceB/RW3/e/YbavbdlXyHvrxEPQ+hJ5udlyj0'
                'tJqc+/9icZQk/XmJ3sH6ZXZeIhfbj+hzaut0I1mLt0EmHt4mnzPDn6MYJ25+'
                'uBpwIvI3NZwYfs7iBEn6cSL0SDXgRAz3H1MvqeFEzcsWJ4JMPJzoe8kMnAjj'
                '+xFxpR+phnm55hE1nHD+bnGCJP3zcg2PmjQvx8OJHjx/0P+aGk40/MPiRJCJ'
                'hxNDr5mKE7WPVQNOJF9Xw4n6DRYnSNKPEz1vmIQTiFDus/nvQ/4tvp8GUbL+'
                'cKO39bxaoj9YnlzOecXfw/OKf8/1fx/F3rRvkTjIj8s8+u508kWdoeOQQtBR'
                'pyxVK/HyfYddpAc8PWs4HF1vPlLCdyzmO47OZMTjb34b6WPiXVJ/+fm+G8cQ'
                'Xz5P8pboOWl83DcbHdt+XHUSL993/2aizFnlz/c9ME+cqQv6hyN7ivlAj2v2'
                'FfPBizQcVFqGsEoj/XEUe/5Jj/QQVWq/hY0L3/C+uJ8bxmXPNLEegJ6mp6vp'
                'X/8Ms/UK0eTzfYv85qFmyuk3z+RmTZ898obiXgDZXg12bwnLt5i5XuArxYfZ'
                'a42ED7PyXg25DzPyoR/LTsfvqZQ7l9ibo9R317gXhoc1xXs1Pkb5EntT+Pso'
                'iDLVuqGkHN4C+YKP/0TlW8f9vUpOb28PmOJeGGqPA1yD33ryXTD+N5+4htu8'
                'LE909OalMzPEMSsrYg8NKwvqPk6x7rN16Vs3/bMOfK1M+yX4+dURqe+X8NsP'
                'CPnXof7I/Otwjsy/DrqP24rqfhR2Pwlrq6N/anueklNy6eXwvcz3lbpvTGar'
                'ZPsE+ftVPq62gO8vbAu8MWHyK3hMd5J6n0R/v7LXqH6lfnmkrTwcKr+1lYdD'
                'tZd3zZKHvy0a3RZ98NgsVPLyfTu7kraqGvJ9pw2rP+3t28rDofb3GCUPIJvv'
                '2+e6Efm+B4zSNzbfd2am2NJHcZncRMyXxGXPpmK+NJQhs+f19OtVxii90i+P'
                'QSsPh4rrbuXhUPlQjJIHm++7ZguxPYR837GtxHyQ7zuxtZgPoiumtpGs12C+'
                '3prKtNcyv5Lw9ui983mGw+gY/Cvtunx1k367kzXQ7hTWdWvmits1zDvEdlBb'
                '103Uqq3rpnasTHuiSvr1asQovdIvj/gmfh6BQZFHcT7Kfjyva/N9I3kkAq0f'
                'xfKoNUwe/vPc0d3QV2Tn0fPc7H4J5xM0npif7ztsWP1pz19v5eFQcdKMkgeb'
                '77t3J3H/D9p3385q84mjdWI+iAMU+4SYD/zOE7uI+WD3Umq+mA/GmdkF5env'
                '6terBqP0Sl++bxfLweb79hhsvm9Esv28PfV067HzTcEifXF6Bv9ZDXF6hvcQ'
                '4yC0m+TuZs8PlZv0x+lZ/EE1xOmp+Q/0wZG9UCmz1wN7W3sdZOLNJzTtpXc+'
                'QV/8hac/qIb4C05EDSf69rI4QZL++Asnf2hS/AXufnC8r7V5fzWcyC6xOBFk'
                '4uFEYn9TceKQf1UDTtQfqIYTQ/tbnCBJP07c869qwIkojmuQPFgNJ0KfsjgR'
                'ZOLhRPpgM3CiON/3Tf9G7djsfN9Nh6jhxNjBFidI0p/vu+EjpF9m5/tOfRp9'
                'cN+hiuOJz1icCDLxcGLw0KnxYwlavu8wvl/B7jRurIrxz2cVxz+uxTWS9I9/'
                'BjeaNP6R4tXnLF4FmbjzX0eaOq7pwz4/Zo9r6j+vaP+PtPafpCkY13wBsZox'
                'rkHkG18q23k8FX+KiP8liyfNxv9a5PrzUcfpp8bWlyn+lyxWHj+23ETjf0F8'
                'KnxOJT6VLP5Xcv37rwjjf82+9Sl8XRZbsdT6U411x/s9XJfH/4L4lkCefL24'
                'gUR8S3cmvua1IE4+eohv6Xjxvbw6m0fwcORLxGXHsS+7bsDy3VPyfah+3rxy'
                'TgZfcP35eMe+93MeC/0DH8v0mx+XnZWvKJajJydPbnCOlC/IHst3krEc65nv'
                'Yb/LN/ao46xZzei3TI5P/yd8p/91IJl94Mufp79gC3THZ42efSF5TNhYWfxL'
                'sMETtal1zHUenyWTSDb+qz/Bjv+CTNzx33F6x39snvHop8XjGMgzPvyWOF45'
                'fIfztpgvisvDJXHNYX9RskXMB3nG+9rU9t0PzVIb30Uk36FxHvUovEJOPc+0'
                'cXS8WU3OtV+242iSpmAeFetXlnpupY6jpTh6isXRIBMPRzMn6cVRffa/7+iq'
                'sP+nKtr/k639J0m//a//okn2n+cf0of7hYOtav4hTcssTgSZeDgx0moqToSO'
                'qQac6F2uhhPuaRYnSNKPE6ljqgEnhvC8wOjpajiRaLc4EWTi4URNzAycKI5z'
                'EFuK3tzsOAeZM9RwIhazOEGS/jgHj2D9MjvOwdiZ6CtqO0ic4MdBbvoq4nPx'
                'sfsHvyhPlqqVePm1k3cjPeDmcX8AXR/7mpgv/Ci6Hv86bc94/ANYb4fjpP7y'
                '82v3dSE+8B9PXoqO0/g49E3bz6lm4uXXrrlYhJ/lz69d2yvGa+g/Nf5YzAd6'
                'HL9dzAcv0vtzs/sJ+uMWjnb7IVql4jobh723U1y/YVyOfl3Ml4+He56a/tWs'
                'M1uvEE0+vzb4d5L+W+DfCUTmdyX95yC/KyYX16mqfyfr2+WTm3gtedy3xwnv'
                '4WPIVUzdz+d4ovldZf5xda7/+WL5+uV3FeXP9ZOvJH+u48l3oYNyvWIqNdcx'
                '4XuJjt0T1lHH6fMvVfTNnKzvMut7LePn59pFpO7rLPIV9fwcPR/Fgv0t+IoC'
                'v58vrlf/8wo8VF2Bf66krgg/UtaPnObj1GUk5EC7UvKT9jlPXWf94ovyME+8'
                '7vk5yFXsFvhNg1xJv2mvDgi/3kn5pR+w/7uk3Qof++X3FX1Qff18BzY/Z0zR'
                '71ym66p+0yyV5vcPJMv7jUmGCzn+xQ5qF0Deffd0qDyHLm5HbhiXuzr8ecBa'
                'pr9Q7nlA3nN7ND8XSH+/csyofqV+eThrrDyodU0rD4dqL3Gz5OE/P9V0JfqK'
                '5vNRycuvHf0+aeeqIb+281Wz6k97+7bycEh51BglDyCbX9vnuhH5tWuN0jc2'
                'v3Y4Kbb0UVxmzxfzJXE5eoGYLw3/ucjseb0pyK9llF5NQX4tKw+Hyq9l5eFQ'
                '8SyMkgebXzt+idgeQn7twZSYD/JrD39DzAdRP0Yuk6zXYL6xyyvTXsv8SpJX'
                '0+vyY9+h/Q/tunx1k3674xpodwrruvEetX2+g9eoresOX6u2rjtyXWXaE1XS'
                'r1eNRumVfnkMnR9keUBrIeY58Lyuza+N5DEcaP0olkfCMHn4z3Nn0tgvuJee'
                '52bX86KM35v5+bWThtWfbnmkrDwcUh49RsmDza899i1x/w/ad+h6tfnEpu+J'
                '+SD+zaDEPxb8zodvEPPB7p6RG8V8MM50v1ue/q5+veo1Sq/05ddOYznY/Noe'
                'g82vjUi233X0R3TrsfNNwSJ98Qnq8XgvST2v0u0Ru++04Ra1fafZH5o9P1Ru'
                '0h+foA/rV5Z6buXivv94MY73NfX0kfv2+Pa69jZrr4NMvPmE/j698wn68kak'
                'zkJvbnbeiOhP1HAidKvFCZL0543YdS1iNSNvBHffON7XOvBTNZxw+y1OBJl4'
                'ODH8U1NxYkmiGnAidYcaTkR+ZnGCJP04MZSoBpzI4LgG2TvVcCL2C4sTQSYe'
                'Tjh3mYETYXy/QjvOnl0N8079v1TDiea7LE6QpH/eKXpONcw7jQxgu3+34nji'
                '1xYngkw8nKi/e2r8WIKWX3tvfD/CbxdHjKNXIE3DtdRvFMc/91hcI6ncuAbt'
                'l/CjuBd7rhkx/kHE5upj49r4xoMqIV/nxOOeoFIpnpR7zxFU3Jv+WQe+phj3'
                'RhZ/Tz2fJcSTAiLj3mCi4t4sdIrj3ixwUD8Dznm/nT/+94nCOTYfoyxGkTv9'
                'qtdJGSSX7L4F/ia1mEXZ835M1r2TOfd66tg553J8rBp/ive8yf5eva78cmOK'
                'YhRJ4kmxuTEHR6dNy5DvUoj7JItvVu4YW2xcqcJ9WJn4xRUS5buFuE0gQzJu'
                'E/CrxBWC35NxhaAO/OIKwfs7PnbLufXCUuI9FeUwTa668F5/fvY+qnZD/Hs2'
                'ttvdC99i6n2idquO+3u23r264bUFTFDvLhMvyuHIMdW64W1F+y3Lnxtx/c+X'
                'pr/Md1D66xd3TKK/Lr0Dw0ySjVcHH7Lj1SCTvnWo4Uw1rEPVPKI2Dht40I7D'
                'SNK/DtX9W5PWoWR2uPdxa4eDTGy+6cyA2J5AvumGDrX9NlEJXxSXo0+o7bep'
                '/auYD/JNu1m1fegRSRwiaBc9ku/Qlx/oovvQk83OD5T9o5qcU0MWz0jSnx9o'
                'BOuX2fmBap9EX9HwlNp6WebPFveCTLz1suanzPCrKF5/6v1dNaw/hf6ihhP9'
                'T1mcIEn/+lPTepPWn3g44eL+Y/RpNZwYecbiRJCJhxPJp83ACVg7IuI83I8j'
                'n1DPMw0nIs+p4cTw0xYnSCo3TkAuImL+9X41/076+sdFPJyI4fmD1N/UcKLm'
                'RYsTQSYeTvT9zQycCOP7Fdpx8++rwU+7+SU1nHBesDhBkn4/7djfTfLTRpTz'
                'I4gV8qoh/4QzN4A/G98fhiovXk0du6sh554sDxhcV815p+7Pw41/8DKOf/AK'
                'iYP8+Mi9r8+gPiTxRWR16OhPlqqVeHm3BxuQHvD0LHsI9p45VML3Bcz3RfC2'
                'oalzRUskEsHlvrjcD5f743IJLg/A5YG4PAiXDbgk/MybfZ82cUI4tUm+nzA2'
                'Ju4YQn8g2T1dyJfPszwq5nNxOXSmGPdglNj4stq64Eibf73oIlke9dFRvC6A'
                'jwf/hY7T+LhpOnpf2y+vTuLlUW/+gNTnieZNZ/kLJZtHPTpX3C6gffcsFPOB'
                'Hg/sIebDL+Zk95na9lhu0h+fsn8WnCGpUvuhbLz97BtiuxzGZcNbYj7Q08a3'
                'xXygf83vmD1e0q9XA0bplX55ZKw8HFIeg1YeDhXfwzB5cOJAzUB4m34Pr7Jw'
                '8qj2zSNxWVceVfi9t2irPV+CYfWnPY/P6/SKAKLgyqPRKHkA2TyqPteNyKPa'
                'ZJS+sXlUY++q+U3Xvy/mS+KyYUzMl8Zl4z9tvz5XcvWq2Si90i+PqJWHQ61r'
                'WXk4pDziRsmDzaM68KHYHkIe1ZqPxHyQRzW8UcwHu8oijnjeDuar3WmVOb8n'
                'Wycc3AS9N8wnuTuiY/CXsfPy1U367U7t++bZncK87sAscbvO7xMOSewELsOb'
                'qa0rRDavTHuiSvr1KmyUXk1BXkij8F1/HtUklofNo4rH4Vv6eQBVqjwQhVxe'
                '/C34Pm8u1osvRMQvyvXYvBgvs4lzmzhIE7Yv/NZl49x4ElzgoFhcmFTjbxXF'
                'GMqcu5aOMUT7mkUv/f0z+Bh8xVRjALHnWV84WSwpiNXDu6/M962OuV7gU6kr'
                'z/+QzLw3wbqCWD2OF6vHq/8FBG+JdVUcX+qWtdTxbte8qxhvipUdy8f6Jfqv'
                'N7Dxovj8/Fho/s8r3IcXHwpIECsN6iJ3zavjuU6h7rzfeHW8U+HcZOoq9dBB'
                'b5OyIOqujvk2Xswplui6jp7NxALLt1NZzKrc+eZFrz+rWFegO6XH+lJpVxON'
                'QQj3IGMQwj1KjEHIxsMj6k4pviNbF6O39bxKyjbC+PtK78eXdb0rvi6zgert'
                'yi8GoddeWBvo1c9s4vmb4FISgzB689KZGeKYtWUlxPCUxSyEOJKqPs+F+/DW'
                'aOuPRVraeFzxWGcK/CDeM6u/7C/D0C5Idins18bd/8L4vZmfR33QsPrT7rdh'
                '5eFQfhBGyYPNo+5uIZ7/gfbdNEfMl8Zl785iPojrVLO9mA/2EYR3EPPBbrRI'
                'rZgP5plTO5Vnvku/XmWN0it9edRHsBxsHnWPweZRRyTbn92wgJ6FsutNwSJ9'
                '8Wab8fym2fFmk3ViHIR2Ux82e32o3KQ/3uw7WL/MiDfLGy8O4H1Nw7uhUmav'
                'o7tbex1k4s0njO2mdz5BH07M3LoacKKvXg0nmhZZnCBJP06kt64GnMjCvtbF'
                'ajiR2sviRJCJhxPhxabiRP821YATQ3ur4URiscUJkvTjxNE11YAToQi26/up'
                '4UT//hYngkw8nGjcz1ScOHl2NeDE2BI1nEjvZ3GCJP04MTa7GnAiciD6iqYG'
                'xfHEJy1OBJl4OBFvmBo/lqDlUS/OdxTaDvEvp55nGq4NfUpx/HOwxTWS9Oc7'
                'Won1y4x8R1K8ci1eBZm481+HmjGuCeP7Ef6En0H8Sep5xtn/zyva/0Ot/SeJ'
                '9WPfzKH2h7jkXh1Pkzz/dNjvQe4P8fYIzCXuI9gf4pN3O+drnuwPb+GiY4j3'
                'yRLt7x8+71LF/Tq+ccSJPSCy/Oyy/Tz8PR+8fQJA5J4OkBO5p8OTqbdnADdf'
                'dp9ALt+Mt08E4r17RO6/wVTq/hvHOXUd3qMhi3Huvy+Oz8/uEWH5cuXQc5tS'
                'ddWzZj677630/TWI+DHZVfbfTGBPR65+vfrYibgXuQcR6n2CexAFe0HYPYLU'
                'fRzFumX35zjuCetoXZlwXcj2KPLtAFtX40Mx7n5RILKuQOa4rnJ7pTx5E7uI'
                'ffeLQjtk6ir32/HhYa4/AO/gtcl9xv/2Iu6J69LF8z/uIlzuSXybU7zfJ9t5'
                '/Au0rG+9sJS9hiXsD0LP/1Pb84p1rLbni9nrWMY9XrL9e+p7uPx0yG8fK+gQ'
                '6IKfDoF++e1j9ep6AfEcT4c8vVlE3BProQt2Pewg/dqHuDdfh9RJf96RpsMQ'
                'nmWp55o6/htaasd/QSbu+O+Lesd/bN740IHicQzkh+h/RRz/Cr5jWMIXxeU8'
                'yT4k2F9U3yzmg7zxTaeqxd1JvKeWVyo1S8imcX0w8tlqWB+s/ZLaODpzjB1H'
                'k6R/ffCRz5q0PijF0eMtjgaZeDgaO86MedRi+//04VVh/09QtP8+sRyCTPrt'
                'f+IIk+w/zz+kCfcL4yeq+YeMnWRxIsjEw4meE83AiQPw/QrtOHUk4n+Sep5p'
                'OOGeooYTIydanCCp3DixN74v4af6OWxhjcaJBJ4X6I2q4UT4NIsTQSYeTgxE'
                'TcWJRxqrASdiy9RwoqbV4gRJ+nEisdwknEDku/7N+i4kr3jx9VL8QsZppuo6'
                'ttL92Liudy98q0zr5Wz8zDp3cvdjY0EW+Hi4XLMCaU0kRuIyP+9E7EzEBw9q'
                'fIjGaUvVTbz89L13IT3g6Vn/r9D1obslfA9gvgdp/ODxD2G9HT2D1F9+PvPM'
                'KsQH/vq956PjND4OX2r7ldVMvHzm9etE/RWws6XmNy/kD2XzmUeuEfePoL8a'
                'TYv5QI9TPxbzwYv03252v0x/nMjQaj9Eq9R+FJv3pn+luH7DuAytFfOBntYm'
                '1PSv/myz9QqRqk9aoeTF4yd9zjgx3sHf2i1c8/e9PfYU8BWU+XfKclLs6fqf'
                'Ly2vAOPXTPmNQ14BoDL7yibq3pxGfWOm6wYsKzUf8+Lz9HFB1rnj9FNj6+nf'
                'f+tCBgvE95u4nyW/Lkvx8ffqY2uC16srz6ePzSsAfo7gr1iGfB2E//8il/0G'
                '+tt8fcSd5HVU7pVHQheo4q/D4aP4M5e8HMqgY7X8Hvz35/ussnXlyYhtV+CD'
                'Du1D4NdM+aCXuB+D3S8RnX3rUxx5y+RX59LH/vLKrIS2yc8xIPp9gfj5Uiay'
                '3wWu+e138cuHAvwS+TrXTp+B33Oxy74n/f6ofax//5VD6e9j+djj1/+TOCb2'
                'UEx2HoK/B6MM+Ja75u2TIXCB2u/iyX6eU9BjUr68cd/YY+gptY+Lx5NjT2K+'
                'P4v7J/r7lTVG9SunIL+klYdD5UW08nCo9nKmWfLwtz+xFJ6V/xoq2fmHPsyf'
                'uIG0Tzx7DfMJ3nwuspelZZsHOx9yp6B9G1Z/2tu3lYdDyqPeKHkArWrvaOmK'
                'r2zvbulYvbytpav9nFxaCAcSUOTKFWs6lrVgfu+4a21797JY7ky8s215+7Ju'
                'OukEOh9v7WxrWd2xsr2jrWVlW2tnR3vH6XDZwfkuxm8UaxlnXdm6rG1VWz4L'
                'xfj17s7W7vYV7ctavfA3LavaumOrvZuPn27P3airrbtlWevKZeTLOMx14r2p'
                '8/HO1fHVnd2FrBjM9bwUvPNtbWSCDKh3mE/ikX59ixilbzCvB/uDGs4R91+j'
                'uBz7mpgvicvQuWK+NC5rzzN7Xk+/XjUYpVf65eFaeTikPBqtPBxq/7BR8kB2'
                'eFY+n1MqKbaHw7jMXiDmG4XyQjFffnfAxZL1GsxXc0ll2muZX0nv5fS6fM23'
                'aX9Puy5f3aTf7jQbaHcK67qpy9T2VWevUFvXHf2m2rquc1Vl2hNV0q9XUaP0'
                'Sr88Rr4WZHkU59usx/O6RCQiJzjyKNaP0UDrR7E8egyTh/889/CN6Cuar6Hn'
                'udn9GwnG7838fOa9htWfbnmkrTwcUh59RsmDzWde0yPu/0H7Dl+tNp8Y6xXz'
                'QbyhrMQ/FvzOR68V8+V3U31LzAfjzObry9Pf1a9X/Ubplb585gNYDjafucdg'
                '85kjku0vDn2Pbj12vilYVK59vthvmVz/wOM9QFDgq2x7xO7zbfq+2j7fsZvM'
                'nh8qN5Ub98H/mog3gvUrST23cnHff7yYwvua+n5I7tvj2+vIzdZeB5l48wmD'
                'P9Q7n6AvvlwojvjNji+XuEUNJ8I/sjhBkv74cjdj/TI7vlw/3tc61KeGE823'
                'WZwIMvFwYrTPVJy486vVgBPpn6jhROOtFidI0o8TTZ3VgBPDOK7B2E/VcCLZ'
                'b3EiyMTDidqfmYETYXw/wg+lS21eoLJxYvAONZyI/8ziBEn68/5ksX5lqeea'
                'hhPO/+JWdZfieOKXFieCTDyccO+aGj+WoOUzD+P7FexOprsacC09oDj++YXF'
                'NZL045q7xiRck+LVry1eBZm48193mzquCZ9VFfb/N4r2/x5r/0nSb//77zXJ'
                '/iPyjd/Vv2lRPDKKnyBZnMDc+fAuiz7E1+tdAV++TK6BONYTj0fFy9kNcblk'
                'eaAxlZwHGq4pxiF03J5rOTEa0fF3d+TnkKfPszm/F7n+fNRx/6wDX1OM/SWL'
                'U8iPreb/+8J9Som95smPjUPoxQPz6g03zcnEtiPiQtZx35co2ViDo/dt3ICP'
                '1fJfJw9ai+tflr9eFrNePXagF0OtlNiBmNjYgRB7DXQdns3G3WfjkBI5x9m4'
                '+ey3yeKU8uPUoxJiearHU4TvF8Sjc8jYl8CvEPuSp3OsfIic7nXM95U7r4As'
                'b4GsTcviMVqqVJKN/9KP2PFfkIk7/ntQ7/iPzWc+/HPxOAbymTfF1PYXJSR8'
                'UVymz1LbXzQo4YN85mN/Vdt33yiJuwTtsU/yHfr8SHoy1eBHMvKompx7Hrbj'
                'aJL0+5Es/q1JfiRSHP2jxdEgEw9HGx43dR71kPuqYR515AlF+/+4tf8k6Z9H'
                'vec+k+ZRpfZ/2Nr/IBPX/v/ZDPu/I75foX3e9DvEv5x6nnH2/y+K9l+S7yFo'
                'VG77j9dHCP1y1iOJQ2ZIRJVq/3n+geH/Q1/hPqPmHzj4rMWJIBMPJ6LPmIET'
                'YXw/Yj18fTWME2qyajgx8IzFCZL0jxMi95s0TuDhROPz6CtiL6jhRPrvGAem'
                '7tUtVRCxOJHP2/sCbX9WtLV2r+lsa4lEIszxvszxfrljwI0w57k8/Q29iPXx'
                '72r6OzaC+OgoppaCQmzed9Dj5HtILzwdO2L56W08fXPHEF/0A1Qe3erlsGlv'
                '7QivXhH+XOuarq7x/+dPLl0BpxCuz8jjdfYFMV7n4xf9XcwHdjj1QTDwX47r'
                '6Hi7acinZhqBxx4R/vKvIont4pBUjN+9TiUT0qtN8vGV3LfU1qsj76v1K0de'
                'FvOFcRl9R8wHOBF/V1Hv355aff5cZ+vy9raO7vAXW0/vaO9es7yNbND5EnPl'
                'mfbP0RO5NHbk/w9vX7GirbOtY1lbwQR0yd7BkiVLlixZsjS1VEq/ckdH1K88'
                'Ffcrd3RIMq1fOfn5M5581r5KrzQgqvR52YnPU/PksA7Lge7nFsuhR88HTZLK'
                'L4/Gv5ssD6CO1lVtXpmLbz5edrWfvqrVO79m1Wltnd6YYr/l4WWx1o6OXFhy'
                '4vz+1PnTO1eviRNxy5s5z5solWLvZjqiejtasd4q297J5g1HPrTzhkEm3rxh'
                'fNqsXCmbN4zMQHxNs1BZmCAg5xm6yDkE3DzV2+lcR9ROE7idzqW+y7R2ys53'
                'Rf6tNt8Vnj5LyAe9kWHJPGoY7rtRbb4r6oifC3rifjS1812l6BXaZ8jTq29i'
                'vdqOur9peqWvv3uh7e96/NMut/1dh5RH8gOT5QEUvP7uRYr1Vtn2TtbfHd4E'
                '4Zbt7waTeP3d6JZIL7y+ro7nltJOd3BE7bQft9MdqPub1k7Z/m44JO5PQn+3'
                'Ziu1/u7gLDFfGJfu5mI+6O82baHW341sJuYrN5VvHHXVq9UwjtLX333gA9vf'
                '9eTwB6P7d+WXxxNGywMoeP3dOEYIs/u7/x8BMXbk')