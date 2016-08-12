'''test_createbatchfiles - test the CreateBatchFiles module
'''

import base64
import os
import sys
import tempfile
import unittest
import zlib
from StringIO import StringIO

from cellprofiler.preferences import set_headless

set_headless()

import cellprofiler.workspace as cpw
import cellprofiler.pipeline as cpp
import cellprofiler.image as cpi
import cellprofiler.module as cpm
import cellprofiler.measurement as cpmeas
import cellprofiler.setting as cps

import cellprofiler.modules.loadimages as LI
import cellprofiler.modules.createbatchfiles as C
import tests.modules as T


class TestCreateBatchFiles(unittest.TestCase):
    def test_01_00_test_load_version_8_please(self):
        self.assertEqual(C.CreateBatchFiles.variable_revision_number, 7)

    def test_01_01_load_matlab(self):
        '''Load a matlab pipeline containing a single CreateBatchFiles module'''
        data = ('eJzzdQzxcXRSMNUzUPB1DNFNy8xJ1VEIyEksScsvyrVSCHAO9/TTUXAuSk0s'
                'SU1RyM+zUggpTVVwLE1XMDRUMLC0Mja0MjVQMDIwsFQgGTAwevryMzAwaDAy'
                'MFTMmRu00e+wgcDeJXlXV2pJMzMrCzdO5WxsOeXgKBLKw7OjQHeazPIuWZff'
                'pTp/xP2Sl/eFqmbOL+l6/2fe+e+PGBl6OBz8639bOWlcMK184DLPP0BTTWwT'
                'N0uf79/w41OW9Zyz8NM8H+DEGb1f/POT1K062+Umne95POOYXmTynKPGPk8f'
                '3+E6+034t6NdYnaLls5byf0zUzoNqor3xev+Vjv12vpCoVROLUfZD4c5Vef2'
                'N+jOv/JjWvKK8Oe6pXO71jG/Xr29N+/KAutVL7pskgxjX8iLBTr+9DKy/qr4'
                'XUDw7ot7C++pl+ft9z//ODqi7sWLadXfdubNSH/xc/Lx3K6Qh7yh8vHlPvu3'
                'Lfv9q9Z5r5Lu3y/3D6dy15d9fS2/QDnqZ3nQi7jvX76uactPfP2/tNF3IQBj'
                'pbBe')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CreateBatchFiles))
        self.assertTrue(module.wants_default_output_directory.value)
        self.assertEqual(len(module.mappings), 1)
        self.assertEqual(module.mappings[0].local_directory.value, 'z:\\')
        self.assertEqual(module.mappings[0].remote_directory.value, '/imaging/analysis')
        self.assertFalse(module.remote_host_is_windows.value)
        self.assertFalse(module.batch_mode.value)

    def test_01_02_load_v1(self):
        '''Load a version 1 pipeline'''
        data = ('eJztVdtOwjAY7pCDxMTond710gsdgwSiuzHDxEgiSGQhmpCYAgWadCvZwYhP'
                '4aWP4SP4SDyCLWwwmoUBemmzZvsP3/+1X5t/dcO8N6qwrGqwbpgXA0IxbFLk'
                'DZhj6dD2zuGNg5GH+5DZOjR9DA1/CItFWNT0UkUvlWFJ067AbkOp1Q/56yML'
                'AH/APp+pIJQJbCUyhd3CnkfsoZsBaXAS+L/5bCOHoC7FbUR97C4pQn/NHjBz'
                'Ml6E6qzvU9xAVjSZj4ZvdbHjPgxCYBBukjdMW+QdS1sI0x7xK3EJswN8UF/2'
                'LniZJ/EKHabKUgclRodcxC/yNbDMT8fk70Xyj7j1zDkF7jIBl1vBze2e3vGw'
                'Nd6IN7WCT4EG22y9m+CyEi4cIS4f0aeZwHcq7VPYnQ5hfWLjDrHQkF+0F2Qj'
                'OnGJG6l7l1D3WKor7EJQrxBT7+sX576NHv+4v8V9gvXnpoDVcxuB9femAlbv'
                'jbB7mNKxw0RfdlRr1jxctTdryV3k9UYi4KrzHl0Vjlsy61nyvvIxfNH1pfhX'
                'LkEPWYelPtPrXfiUGL6DBFw6+EMI3NOW+p+tyQdS/g+XWIS0')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CreateBatchFiles))
        self.assertTrue(module.wants_default_output_directory.value)
        self.assertEqual(len(module.mappings), 1)
        self.assertEqual(module.mappings[0].local_directory.value, '\\\\iodine\\imaging_analysis')
        self.assertEqual(module.mappings[0].remote_directory.value, '/imaging/analysis')
        self.assertFalse(module.remote_host_is_windows.value)
        self.assertFalse(module.batch_mode.value)

    def test_01_03_load_v2(self):
        '''Load a version 2 pipeline'''
        data = ('eJztVdFOIjEULSMaxcTsJj6sb330QYdZsuwGXlYwGkkEiRCzD5NoBy7QpNOS'
                'mWLUL/Fz/CQ/wXaYgaEhjOK+7TaZdG57zzntmXZus9a9rNVx2XZws9Y9HlAG'
                'uM2IHIjAr2Iuj/BpAERCHwtexecBxbXJEH//gZ1KtfSrWqrgkuNU0Hot12ju'
                'qW60jdCW6lWHrHhqM45zqUfHHZCS8mG4ifLoWzz+op4bElDiMbghbALhXCIZ'
                'b/CB6D6OZ1NN0Z8waBE/naxaa+J7EIRXgwQYT7fpA7AOfQJjC0naNdzTkAoe'
                '42N+c3SmK6Shq33AG3Mfckt8KKTGdb6D5vn5JflWKv+LilpiirvLwNkLuGns'
                'ulT0KQeX+mSoPsAt4YQ9hjR02yDGDNxLANcjsjdSO5OfW9/fwG0ZuKQluJ2U'
                'j6MMvbLhR/m9fpw9EF+9d+qdhsqKvrjWa2foHRh6B6v0Uvu4yOD9avDquBjz'
                'Fdfg2zf49qN1cioDMQTuej3vVrNDwneSwVcw+HRcVCzFGcuU54+1/j35yLn4'
                'j/s3cc9o9fnKocXzlfX/+IkWz7WOe8DYOBC63ga2HxWF0O5FpTb6i+qJ0J7W'
                '3roeOKdRLTL3tbNEL70+S70VMvwwfZj78/p7HT1rid5uBi4fV/7ofn/Q/8MV'
                '+cjIfwPWKbcg')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CreateBatchFiles))
        self.assertFalse(module.wants_default_output_directory.value)
        self.assertEqual(len(module.mappings), 2)
        self.assertEqual(module.mappings[0].local_directory.value,
                         '\\\\iodine\\imaging_analysis')
        self.assertEqual(module.mappings[0].remote_directory.value,
                         '/imaging/analysis')
        self.assertEqual(module.mappings[1].local_directory.value,
                         '\\\\nitrogen\\bcb_image')
        self.assertEqual(module.mappings[1].remote_directory.value,
                         '/bcb/image')
        self.assertFalse(module.remote_host_is_windows.value)
        self.assertFalse(module.batch_mode.value)

    def test_01_04_load_v3(self):
        data = ('eJztVtFu2jAUNTSt1iJNm9SH9c2PfdhCSsu08bLB1KpIhaIFVX2I1DpwAUuO'
                'jRIztfuKPvZz9hn7jH5C7ZBAsBApbI+1FDnXvufce0+c3LTq3Yt6A1dtB7fq'
                '3U8DygB3GJEDEQY1zOVH/CMEIqGPBa/h7miCXRjjygk+Oq5VnNqxgyuO8xVt'
                'NgrN1ls1/X2D0I6a1YSKydZ2Yhcyl7ZdkJLyYbSNLPQhWf+jrisSUuIzuCJs'
                'AtE8RLre5APRvR/PtlqiP2HQJkHWWY32JPAhjC4HKTDZ7tA7YC79DUYJqdtP'
                '+EUjKniCT/jN1VlcIY24WofrrbkOhSU6lDLr2t9Bc39riX8x4/9OWW0xxd3m'
                '4OwF3NT2PCr6lINHAzJUD+CGcMLuIxp5HRBjBt4FgOcT2RupyuS/5fc/cDsG'
                'Lh0pbjej4ygnXtXQo/pSPU7vSKDu3YbbVF7xE39JfdZCPAt9cY5ONK6Tgzsw'
                '8jxYlWem/vMc3vcGr7bLCV95A759g28/zpNTGYohcM/v+TeaHVK+7zl8ewaf'
                'tsuKpTxjmfI8FDd/v9Y5T6+4V9w6uEe0+lwW0OK5zPtefUaL74O2e8DYOBS6'
                'v4d2EDehyO7FrT3+auuNyJ72+oZeOKNx7zPr2l0SL5tfUd2VcvQwdZjr8/Rt'
                'k3hbS+KVcnBW8qcR99019T9c4Y8M/2eDnrm5')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        self.assertTrue(isinstance(module, C.CreateBatchFiles))
        self.assertFalse(module.wants_default_output_directory.value)
        self.assertEqual(len(module.mappings), 2)
        self.assertEqual(module.mappings[0].local_directory.value,
                         '\\\\iodine\\imaging_analysis')
        self.assertEqual(module.mappings[0].remote_directory.value,
                         '/imaging/analysis')
        self.assertEqual(module.mappings[1].local_directory.value,
                         '\\\\nitrogen\\bcb_image')
        self.assertEqual(module.mappings[1].remote_directory.value,
                         '/bcb/image')
        self.assertFalse(module.remote_host_is_windows.value)
        self.assertFalse(module.batch_mode.value)
        self.assertEqual(module.revision.value, 8014)

    def test_01_05_load_v3_batch_data(self):
        '''Test loading a version 3 pipeline with batch data in it'''
        data = ('eJztnX9vG0d6xynHzsVO2kuCAtf+EYAHFHDSRvLO71m3yEm2klio5egi937g'
                'fPXR0lpiQZECSSVWiwPuz76k/tnX0lfQl9CdHe5wnglHpKhHEimNAFneJZ/5'
                'Pt9nd4ef3VnObm+8fL7xpCnWsub2xsvVt+1O0dzptIZve/2jx83u8Mvm037R'
                'Ghb7zV73cfPl4UlztzhuUt4k7LGgj7ls0izLG/P9rGxt/3X5578PVhrvl38/'
                'KH/vjF66N1pe8X7N8m4xHLa7B4N7jbuNvx2t/5/y9zetfrv1plP8ptU5KQZj'
                'iXr9Vvdt7+XpsXtpu7d/0iletI78N5c/L06O3hT9wXdv68DRyzvtd0Vnt/0f'
                'RWChftv3xQ/tQbvXHcWP2g/XOt3eMNA1dTj834arw0pQh0/L38+89eb9zxrj'
                '99+dULdPvPd/PFpud/fbP7T3T1qdZvuodeCymKW9vwnaM8ubxdvWSWdoG2u+'
                '7XX2i37dnp7S3vtBe2a53/rx2292LhS/+WJjtvgPgniz/LTX7z8pdyETvz4l'
                '/n4Qf38U/22/KLpeXefx8eJkr1O0ceKn+firIN4sb7YHw1Z3r3hadDqDxmzt'
                'TKrHTr93XLdR1yOb0s4KaGelQZHqOE/+k7bnefMnF8j/+9aw3SMXjKezxd8L'
                '4s3y9unur5/P6Ps9EP9e4/dl7zJL3T8OdM3vy+LdcPXrd629YfOoNdw7nKfu'
                'aw3c7X7R/uSy99+JdTzsF4ND0y3vV4ch5nE8rZ1Z+5Xr7h/Pu1/JGf3Puj0u'
                'ul/Nm/88dds09FDgxM/DG8+L4nXZLRwdd4rdJ7tbDmFMe7+b0t7nQXtm+ZEB'
                'lxIoH7W6rc7poD14tFP0ysYflUKPjk/fmL6nZLXhPHVmjfk57elhq9stOmTr'
                '+fN/3V4ru8DR6/PuL1udzslRSVYXii/JapY63AXxdxv6S6rn3V/Yl3n5M9vx'
                '9mEQb5ZNv9U6MB+CXVe/ae18FLRjlut+q7nafDFrO7H+7+npsNzFWoOjC7RT'
                '9cfnbGfWz7l52pnn8+FB0I5Z3uw1u71h82RQjNu5St767s2/F3tDnPhped8B'
                '8XcaL3oXi8Oq92X3L5f9uTIvp14W317mcTdPO7PuB1jtXPZ5+FVx5rQ8rup6'
                'wLz1CD8v5jkf/Prd8fB143KPs1j/dgGOopM4alodJ3HAi16zBPi9VqdkRux2'
                'Fu3zYhJPfTccnDS/7fTetDoz13HWdq7a/7T9/2dB3mbZfr7P1u9j8R/WeeS0'
                'fmXa+dPDIN4sb3WHRXfQHp6+Nv876JuhgvG6ce83S/u/DNr/JWh/u2h1J7Vs'
                'f+Y5L9n99fPXVb9yMOU89CA4Dy2Xt3L5YO3z/WPyYPdht7q6//DBMX2wlT0Y'
                'TFpNJq+mk1ezyav55NVi8mo5ebWavFpPXp1H7MRsRnySiFEScUoiVknEK4mY'
                'JRG3JGKXRPzSiF8a264RvzTil0b80ohfGvFLI35pxC+N+GURvyzil8V25Ihf'
                'FvHLIn5ZxC+L+GURvyzil0f88ohfHvHLY0duxC+P+OURvzzil0f88ohfEfEr'
                'In5FxK+I+BWxririV0T8iohfEfErIn5lxK+M+JURvzLiV0b8yljfHPErI35l'
                'xK+M+FURvyriV0X8qohfFfGrIn5V7MMo4ldF/KqIXx3xqyN+dcSvjvjVEb86'
                '4ldH/OrYp2/Er474zSN+84jfPOI3j/jNI35zz+/zXmvfXn9/TKpXy/XMAFD5'
                'l5ev25H88hXx4PPOsQQR1b87/Z65ZmOaVg9aJSK1dh+eeTl+dN3fNvEoHAUo'
                'm9GmifrK+WpGVjfKf9aG7bfla/mD1qBKavPFRrlYEkyZVckrB0jaJeKMxWkg'
                'XnJOazAoZUzt+IODqiQlzKCJS+icGnFai5cig4ORYY2nmUPDQLMEJmO42ilK'
                'SLKGSyrCEi9BChhmRpzV4nxsmOIVmUpoGGqqyjCrDOvacI4mXhIZMMyNOB+J'
                'l1jmDDO8IjMGDUNNXhnm1WEvRoZLTkMTV9CwMOKiFteeYbwi8wwaBpqcVIZF'
                '1b/RkeES+NDEOTQsjbisxcXYMMcrMlfQMNTUlWFZGc5HhktyxBIXBBpWRlyN'
                'xEvidIYFXpEFh4ahpqgMK2O4JNKRYYUnHnw+aSOua/F8bFjiFVkSaBhoSloZ'
                '1sZwibbWcMmyaOICGs6NeF6LS88wXpGlhoahZl4Zzo3hkpGtYYUHAYoCwyQr'
                'xUk2EldsbFjhFVkJYDjQlBY8KioraXvkGI8CVA4dG+whNfbobOxY41VZQ9QK'
                'NEeoVbGWrllL42GAhqxFDPeQmnu0x1oar8oashbUzC1rkQq28hq2cjwOyCFs'
                'Ebb6xONqD7ZyvCrnELYCTQtbpKKtvKatHA8ESAZxi3Aj74A683iLZHiFJhkk'
                'rlDWIhepmItkNXSRDA8ISAaxiwiTAXMZaN84Yr0JJK9Allj0IhV7EVLDFyF4'
                'YEAIxC8iTQbcZeDxFyGI9SaQwEJZi2BEWuM1hBGKBwiEQgwjymRQQy+hHocR'
                'ilhvCkkslLUoRioWI1Q643igQCjEMaJNBtJl4PEYYYj1ZpDIAllmkYxUTEZY'
                'DWWE4QEDYRDLSG4yUC4D6RtHrDeDZBbKWjQjub1GUsMZ4YiXaDjEM5qZDGoY'
                'JtzjM8IR680hoYWyFtFoZo0rZxzxUg2HlEaJyaCGYiI8TCMCsd4Cglooa0mN'
                '2qtiYnxZDPG6mICwRmmZgYNjIvwrYwKx3gLyWiArRxfHKmAjsiY2IhFJQkJm'
                'o4afHCMT6UEbkZjXISG2hbKW2yizxrUzjkgSCpIbNQhFxpdCfXJTiPVWkNxC'
                'WUtu1JKbcuSmEElCQXKjYvWpdwFa+eSmEOutIbkFstqSG7Xkph25aUSS0JDc'
                'qDQZuNJrn9w0Yr01JLdQ1pIbteSmHbnliCSRQ3KjymTgoDn3yS1HrHcOyS2U'
                'teRGLbnljtxyRJLIIblRbTJw0Jx75EYzvHrTDJIblKWZJTdakRvNanKjGR5J'
                '0AySG81NBsJlIH3jePWmGSS3UNaSG83tYE9NbhRxcI0SSG4sMxnU0EyJR26U'
                'INabQHILZS25scwaV844HklQAsmNEZNBDc2UeuRGKWK9KSS3UNaSG6vIjVLu'
                'xvcQB9soJDdGTQbaZaB844j1ppDcAllmyY3ZcU3mBjYRB90og+TGmMkgdxn4'
                'Y5sMsd4MklsoOxretOObzA1wIg6+UQ7JjfEyAwfNlHvkRjlivTkkt1DWkhur'
                'yI3ymtwo4iAc5ZDcmEEoB82Ua984Yr0FJLdAVlhyYxW5UVGTG0UcjKMCkhsz'
                'COWgmQqP3KhArLeA5BbKWnJj0hqvyY0iDspRCcmNqdXNMTRT6ZEblYj1lpDc'
                'QllLbqwiNyqlM45IEhKSG9MmA1d66ZObQqy3guQWyCpLbsySm3LkhjhIRxUk'
                'N5abDMY3j/jkphDrrSC5hbKW3JglN+3IDXGsjmpIbjwzGTho1j65acR6a0hu'
                'oawlN27JTTtyQxyyoxqSGycmAwfNuU9uOWK9c0huoawlN27JLXfkhjhyR3NI'
                'bpyaDBw05z655Yj1ziG5QVmWWXLj9jbFrCY3hjh6xzJIbpyZDJTLwCM3luHV'
                'm2WQ3EJZS26cWePaGUe8N41AcuPcZFBDMyP+3WkEsd4EklsoO7pBzd6hRtwt'
                'aoijd4xAcuPCZJC7DPy71AhivSkkt0CWWnLjFbkxWpMbQxy9YxSSG5dlBg6a'
                'GfXIjVHEelNIbqGsJTcurfGa3Bji6B1jkNy4QSgHzYx55MYYYr0ZJLdQ1pIb'
                'r8iNMemM45EEY5DcuEEoB82MeeTGOGK9OSS3QJZbcuMVuTFekxtDHL1jHJIb'
                'z1e/HkMz49I3jlhvDsktlLXkxnN7+21Nbgxx9I4JSG4iMxm40guP3JhArLeA'
                '5BbKWnITmTWunHE8kmACkpsgJoMampn0yI1JxHpLSG6hrCU3UZEbkzW5McTR'
                'OyYhuQlqMnA3fEvlG0est4TkFsgqS27Ckpty5IY4escUJDfBTAbuZnPlk5tC'
                'rLeC5BbKWnITltyUIzfE0TumIbkJbjJw0Kx9ctOI9daQ3EJZS27Ckpt25IY4'
                'esc0JDchTAYOmrVPbhqx3jkkt0A2H33FwJJb7sgNcfSO5ZDchDQZOGjOfXLL'
                'EeudQ3ILZS25CUtueU1uHHH0jmeQ3IQyGdTQzDOP3HiG+K2ODJJbKGvJTVTk'
                'xjPpjOORBM8guQldZuCgmWceuXGCWG8CyS2QJZbcREVunNTkxhFH7ziB5CYM'
                'Qjlo5kT6xhHrTSC5hbKW3ERuv0dUkxtHHL3jFJKbNAjloJlTj9w4Raw3heQW'
                'ylpyk/YLolQ543gkwSkkN0lWvxlDM2ceuXGGWG8GyS2UteQmK3LjrCY3jjh6'
                'xxkkN0lNBq70TPnGEevNILkFstySm6zIjfOa3Dji6B3nkNwkMxkwl4FHbpwj'
                '1ptDcgtlLblJZo1rZxzxq4oCkpvkJoMamrnwyI0LxHoLSG6hrCU3WZEbFzW5'
                'ccTROy4guUlhMnBf0hTaN45YbwnJLZCVltyk/XaodF8PRRy94xKSm5QmA/dl'
                'TQm+IYpYbwnJLZQdfUnUfktUOnJDHL3jCpKbVCaDGpq58slNIdZbQXILZS25'
                'SUtuypEb4ugdV5DcpDYZaJeBT24asd4aklsgqy25SUtu2pEb4ugd15DcZG4y'
                'cNCsfXLTiPXWkNxCWUtu0pJb7sgNcfSO55DcVFZmMIbm3Ce3HLHeOSS3UNaS'
                'm7LkljtyQxy94zkkN2UQykGzyDxyExlevUUGyS2UteSmKnITWU1uAnH0TmSQ'
                '3JRBKAfNIlO+cbx6iwySWyBLLLmpitwEqclNII7eCQLJTbHVb8fQLIhHbgJx'
                'phRBILmFspbcFLPGtTOORxKCQnJT3GTgSk89chOIM6YICsktlLXkpipyE7Qm'
                'N4E4eicoJDclTAbMZaB944j1ZpDcAllmyU1V5CZYTW4CcfROMEhuSpoMuMvA'
                'IzeBOIOKYJDcQllLbkpa426CD8TRO8EhuSllMqihWXB/jg/EmVQEh+QWyo6m'
                '+bDzfHA30Qfi6J3gkNyUNhlIl4FHbgJzRhUByS2QFZbcVEVuQtTkJhBH74SA'
                '5KZyk8F4ghXpG0est4DkFspaclO5ndmlJjeBOHonJCQ3nZkM3OQu0iM3gTjD'
                'ipCQ3EJZS246s8aVM45IEhKSmyYmgxqahfLJDXGmFaEguYWylty0JTflyA1x'
                '9E4oSG6alhk4aBbKJzfECVeEguQWyGpLbtqSm3bkhjh6JzQkN20QagzN2ic3'
                'xHlXhIbkFspactOW3LQjN8TRO5FDctMGocbQnPvkhjj9isghuYWylty0Jbfc'
                'kRvi6J3IIblpsfrMg+bcJzfEWVhkBskNysrMkpuuyE1mNblJxNE7mUFy09Jk'
                'QF0GHrlJxFlYZAbJLZS15KalNV6Tm0QcvZMEkptWJoMamiXxyE0izsIiCSS3'
                'UNaSm67ITRLpjCNOH0YguWltMuAuA3+WNsRZWCSF5BbI0tFEbXamNuqmakMc'
                'vZMUkpvOTQbCZeDP1oY4C4ukkNxCWUtuOrdT1NXkJhFH7ySD5JZnJoMamiXz'
                'yE0izsIiGSS3UNaSW55Z48oZxyMJySC55cRkUEOz5B65ScRZWCSH5BbKWnLL'
                'K3KTvCY3iTh6Jzkkt5yaDNykiFz5xhHrzSG5BbLCkltekZsUNblJxNE7KSC5'
                '5cxk4CZHFB65ScRZWKSA5BbKWnLLmTWunXFEkpCQ3HJeZuCgWUqP3CTiLCxS'
                'QnILZS255RW5SVmTm0QcvZMSkltuEMpBs5TaN45YbwXJLZBVltxyS27KkRvi'
                '6J1UkNxyg1AOmqXyyQ1xFhapILmFshW5DXYf7rSGh93WUeHmI5flad3uw1fl'
                'T7u33+4W5V+bx+s6j1evbCKvXpWZvHoFUnGLIBczKz6QsvOxS03wpcwE8/Vz'
                'eKY9p+sXwXN+zHL9HPmqyeZmu1/sDXv904s9j3Dzp88rfT+Ir3/q+PuR5wnN'
                'G3cVzx88bxxZy9DirqKe1/G8PKznq87zXNRZn/d1ndv9PPvn8ZS4fwr8muUL'
                'dMOuztd53F9F3Hrj7LpOej7ndu9b/7F7cx9f263uiWlj3v45W8vol+WONe9z'
                'Hcv46se0Uf0g1POytsOs/cJlPcdx1uejXpWPq+6XVkDcSmNtxjx/8hzzjHCs'
                'z+lF7lfOvx3opW6/cDuUxz7BiluE/mHW4/Oq9TGeU4/1HOZFeC77zpS4vwt0'
                'zbI7yQrPsbztc9P7kzn5gF8FH8yy/UmGF7cI/c085xeX3d/Mc7512f3WeZ+X'
                'Pu089Cqfe/6T86fR61jP2V7k/ua8/Wk2I78gcojAiluE/mTW43lZ877u56Jf'
                'tP15nnPfPjrYf5ORaz9+FzluvXF2XSedL7ttdA3tTLoetNlrdnvD5smgwG9n'
                '0bbHrJ/zi/Z5FV6/IDPGYV9nmYd/9o7LXaJ/9ceZG/jz2kn9ynztnJ+n5tvP'
                'JsUt8nZa9H6Czxi3EP3EzsmAXbCfmEfXjopf//a8zP1AzBg373WP8DzlRa9b'
                'YMUty3ZYlrh5z/cxx6M+asDj0CyP7v/YfLK2/+ay/c7CU8u8PUN/6pLjsPef'
                'ZdwOyxK3COOxl7ndb/rnzFXcT7Y+Je5DEGeXX/7Ya+6V59AD7/4brHaWcTst'
                'S9xN7A/+MkXvXxpwvzPL//b5r3b+eafTGhZfrf3jF6/N0m+LTuf73o9f/WFj'
                'deePX9RrnvY6J0fdr/6QreZ//E/yJf2zffNuu4ysVn4xsc6LvD+sT6nXVV83'
                'fDalnUnjXb8t2geHw2K/+UOr3zY3HDQur71l2a4p7vLOh86jdzhFTzfg/meW'
                'TZ/y+6LVH3U0/M9frJpV273u8HC0jo7WbbZOx2uuu56LwDXznpdO6p++6fWL'
                'g37vpLs/zhurnWXcTinu+uLWG2fvd2eMg9BlHAe5rOsOy34/XYq7nXFzjm9s'
                'vthYLp9Xfb1jGeuS4m5n3CzX7ekC5HmZx/us9xdMuH+SXbe/FJfiLno83PTx'
                'jHQ/QYq7TXF/aZy9X6fxiuu9n2UZ65Libk/cYePs/TqNL6TrCCkuxd2muNmu'
                'Iwi0uGWpS4pLcTclbt2LS/cpprgUd7vjnnlx6b7BFJfiUtx5eCHd/5fiUtzt'
                'jEvXB1Ncirs9cZjzJswStyx1SXG3J+6DH8dxK0Hcp+XvZ976ZeF1M+1lBdrH'
                'qZ1J1y171TzI44ZucjtXeV15Hh+zbudl6U9uetx64+zt+XEDbk/zO96ezXZ3'
                'vzjGbOfZlHYmXZ/09lPbkOfvtrU3bTuk/ir1V8sct97A2Z5Y7SxL3eaNW/b+'
                '7qq5c1m2602PW2+cvT1n5ZFl8XtT+tNp/cOnQTtmuXcy7LS7xU86iMto76rq'
                'tWjnp8vqa1mOwxSX4hbx/Cgdr4sRt944u76JZ25H3HpjsY7XRWtnWbZjiktx'
                'y9RfXLf+dftIcSnuJsatN673uMLSv24fKS7F3cS49cb1HldY+je1nWXZj1Lc'
                '7Yxbbyznfr6seae45Y571jh7v7vu8c90HKa4FJfiUlyKS+OW1x+33ji7vul+'
                '2BSX4lJciktxKS7FpbgUl+Lw4p55cel6XYpLcSkuxaW4FJfiUlyKu+1x615c'
                'uv6f4lJciktxKS7FpbhljFv34tJ4e4pLcSkuxaW4FJfiUlyKS3EpLsWluBSX'
                '4lJciktxKS7FpbgUl+JSXIpLcSkuxaW4FJfiUlyKu+y4ww/GcStBnFn2v09m'
                '3v8nT2fSfSn/4L3/49HyXtHpHPd7b9udor921Ns/6RSDtU6vtV/dsDJYe17+'
                'd6v6r5fXFB0e6PCzdAbt7kGnqNQqsd1quZKs9d5N0XsW6D2L6e31+v1ib9ju'
                'dE6O2t3WsN3rvm4dH3dO157aV7b8VzbMK4ugfzxFfz3QX4/pt/eL7rD99vS4'
                'X9b7ZNg7KoX21rZGa3fKtRv12uvUnbZ/6UBXT9MdFHu97n6rf+o0d+s116GH'
                'tj/VesOiP2yXjQ9O3vSLg3LfcbovR6/s1q8sgv60/epxoP84pn9UtAYn5XFj'
                'Dp5OddCsbdtVT8erbDuz+H4S6D6Zomvnqmv1y6XD1nFRa39Xrd4oV++a1Zet'
                '3+4Oi+6gPTyF+lv16rH+tP2cBvp0iv6weDcs/9S6L+3iWO9PU/SyQC+L6e21'
                'Onsn5dYsyi7icO1pvbRdLl2lzrT6yUBHxnSKd8e9/nDY228NW29ag2Lt62rF'
                'y97maMVs22tmvb1yXxwWb1rDvUPzwmDtabXiiVnxjVlh9Z55evcn6Pm8cWe0'
                '/Iv379377LNPPrl378Of3f/o5z//+P54+8e4p+G1c7/hc9D//WrePO6urKzc'
                'vXvnTvlnxfzzwXvjPH7ntffhlPZMnf++MfuPaf+/3jsfp33eiL+/MVqX3p/e'
                'n95//vf/P0+1wjU=')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, C.CreateBatchFiles))
        batch_data = zlib.decompress(module.batch_state)
        image_set_list = cpi.ImageSetList()
        image_set_list.load_state(batch_data)
        self.assertEqual(image_set_list.count(), 96)
        self.assertEqual(image_set_list.legacy_fields['PathnamerawDNA'],
                         '\\\\iodine\\imaging_analysis\\People\\Lee\\ExampleImages\\ExampleSBSImages')

    def test_01_06_load_v4(self):
        data = ('eJztWt1u2zYUlhMna7Zha3eTXepqaIpFpeLUi40hjRPXm4f4Z7XRom0CjbZo'
                'mxgtChKVxS0K7DH2GLvcI+xx+ggjFTmSmSyyldhuAQkW5EPyO+fj4TmkRbNW'
                'ah+XDtUnGlBrpfZ2DxOkNglkPeoMi6rFvlePHAQZMlVqFdWX/NnoMlXfVXW9'
                'yD87u+oOAAUl2ZWp1r7ijz+/U5R1/rzH75Wgai2QM5FbyC3EGLb67pqSVb4N'
                'yj/w+wV0MOwQ9AISD7mhiXF51erR9si+rKpR0yOoDofRxvyqe8MOctxGbwwM'
                'qpv4HJEWfoukLoybPUdn2MXUCvCBfrn00i5lkt3WgP5RcTgdSf8hZN1Bi/ER'
                'mCwXfvv3Qei3jOS3VX5vRspF+5+VsH32Gj8/iLS/H8jYMvEZNj1IVDyE/UvW'
                'Qt9ejL41SZ+Qa6PWr8cBHsTgVyfwq8orbl3gDmJw9yW74m6jc7b97Bzy4B0K'
                'l07jj01Jj5B5tP9ggLyxkzfqyHOoaw+QE/D6K0YfkvQJ+XXxJKLyF9ilnaOB'
                'Z1r0zP19dFL1XS5aFAwdGHrOOKJDmyCGjArIgT3jp3YTusiAlmm0HWi5XQfb'
                'jAecUeE9pc5Jw2O2x6b0d2aCX0bJ3XKc4nArE7gVpU6Vqcb3c8mPQi5T1aJM'
                '9dwgT5LqqfAJ0BKzghHqSdqPecX3tLz/jtHzVtIj5DnHY6CgWaqXG89LBijo'
                'AOT0HAAgH/JOkpdl1IMeYapvQC1jBwl7o7nGb3YCl1X29L3dJHmmKcnyc1qc'
                'HJ+vi1dx6xJufI1xG8HzNvmQZP15LFYdvuA/hhYkIxdH15955eU0uEX4K259'
                '/Uzyl5DxsG92gH5nfGfBxfFdl/gKuWvzKdtZvF0TaO1q5Ra8m56bWwLvhoP7'
                'JphufftSwgs5mCfLh5rZGfNYlt8XnYdx/tqQ+ApZrKuqFf4uXwZvO8ZeQeIt'
                '5IdPmz+K9zi0rz3aMoT0EhGy/6a03Tx9A7YLp+923m/1gKioYETMfV6YO90S'
                'obW0fs663taphZKs7yVCFh/3+vVx/7HOy/48o386fJfp34MYvknmlUXwXuS8'
                'oi+xn3cxryyC529fz7afs0w/prgUl+JSXIpLcSluPriDCG7a/zn8P2n6DvVs'
                'FVsmsj+l/qa4FJfiQtxBBJfmf4pLcXe075EJcfJ7trxf6L+XKzfn4SNlMg+F'
                '3EWE2A4V52kcbegf+nA1QqF5cYpCO+Zfq5EDFcLOIMZOXrKT/z876NymDmPU'
                'hAx2oIu0Z35Bm5aDgju21/WPCHXEuQpR4WoXZ4b8sytio8u9Ol4b19iL+n2F'
                'S5vf3LtxnOXxDcf9w9Mk9rIb2Sv7j1/E4LIRTuIS+H+U2eLr4Q3tx31cZPtZ'
                '/aZkMrfud2gne8npQv9i2v8HD8C1zQ==')
        pipeline = cpp.Pipeline()

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))

        pipeline.add_listener(callback)
        pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
        self.assertEqual(len(pipeline.modules()), 3)
        module = pipeline.modules()[-1]
        self.assertTrue(isinstance(module, C.CreateBatchFiles))
        self.assertEqual(len(module.mappings), 1)
        self.assertEqual(module.mappings[0].local_directory.value, 'Z:')
        self.assertEqual(module.mappings[0].remote_directory.value, '/imaging/analysis')

    def test_01_07_load_v7(self):
        data = r"""CellProfiler Pipeline: http://www.cellprofiler.org
Version:3
DateRevision:20150713184605
GitHash:2f7b3b9
ModuleCount:1
HasImagePlaneDetails:False

CreateBatchFiles:[module_num:19|svn_version:\'Unknown\'|variable_revision_number:7|show_window:False|notes:\x5B\x5D|batch_state:array(\x5B\x5D, dtype=uint8)|enabled:True|wants_pause:False]
    Store batch files in default output folder?:Yes
    Output folder path:C\x3A\\\\foo\\\\bar
    Are the cluster computers running Windows?:No
    Hidden\x3A in batch mode:No
    Hidden\x3A in distributed mode:No
    Hidden\x3A default input folder at time of save:C\x3A\\\\bar\\\\baz
    Hidden\x3A revision number:0
    Hidden\x3A from old matlab:No
    Launch BatchProfiler:Yes
    Local root path:\\\\\\\\argon-cifs\\\\imaging_docs
    Cluster root path:/imaging/docs
"""
        pipeline = cpp.Pipeline()
        pipeline.loadtxt(StringIO(data))
        self.assertEqual(len(pipeline.modules()), 1)
        module = pipeline.modules()[0]
        assert isinstance(module, C.CreateBatchFiles)
        self.assertTrue(module.wants_default_output_directory)
        self.assertEqual(module.custom_output_directory, r"C:\foo\bar")
        self.assertFalse(module.remote_host_is_windows)
        self.assertFalse(module.distributed_mode)
        self.assertEqual(module.default_image_directory, r"C:\bar\baz")
        self.assertEqual(module.revision, 0)
        self.assertFalse(module.from_old_matlab)
        self.assertTrue(module.go_to_website)
        self.assertEqual(len(module.mappings), 1)
        mapping = module.mappings[0]
        self.assertEqual(mapping.local_directory, r"\\argon-cifs\imaging_docs")
        self.assertEqual(mapping.remote_directory, r"/imaging/docs")

    def test_02_01_module_must_be_last(self):
        '''Make sure that the pipeline is invalid if CreateBatchFiles is not last'''
        #
        # First, make sure that a naked CPModule tests valid
        #
        pipeline = cpp.Pipeline()
        module = cpm.Module()
        module.module_num = len(pipeline.modules()) + 1
        pipeline.add_module(module)
        pipeline.test_valid()
        #
        # Make sure that CreateBatchFiles on its own tests valid
        #
        pipeline = cpp.Pipeline()
        module = C.CreateBatchFiles()
        module.module_num = len(pipeline.modules()) + 1
        pipeline.add_module(module)
        pipeline.test_valid()

        module = cpm.Module()
        module.module_num = len(pipeline.modules()) + 1
        pipeline.add_module(module)
        self.assertRaises(cps.ValidationError, pipeline.test_valid)

    def test_03_01_save_and_load(self):
        '''Save a pipeline to batch data, open it to check and load it'''
        data = ('eJztWW1PGkEQXhC1WtPYTzb9tB+llROoGiWNgi9NSYUSIbZGbbvCApvu7ZJ7'
                'UWlj0o/9Wf1J/QndxTs4tsoBRS3JHbkcMzfPPDOzs8uxl8uU9jPbcFWLw1ym'
                'FKsSimGBIqvKDT0FmbUEdwyMLFyBnKVgycYwY9dgIgET8dTqRmolCZPx+AYY'
                '7ghlc0/EJf4cgClxfSTOsHNr0pFDnlPKRWxZhNXMSRABzxz9L3EeIoOgM4oP'
                'EbWx2aFw9VlW5aVmo30rxys2xXmke43Fkbf1M2yY76su0LldIJeYFsk3rKTg'
                'mh3gc2ISzhy841/Vtnm5pfDKOhTmOnUIKXWQdVnw6KX9W9Cxj9xQt6ce+3lH'
                'JqxCzknFRhQSHdXaUbTGwcffRJe/CXAk0BKX9sHNK3HIs4QvrdjeJSpbUEdW'
                'uS79rPv4mVb8SLmcOrGw3ugr/lAXPgRe9Zl3uAsXBnkO+sp7VolXyrscMm5B'
                '28T91/02/lHgphSce7i4GdCJ0y/fOSVfKe9RE1/UsYE1TXP91H38rCh+pCzG'
                'uYwpbRhcLlHGiWXY7OuJaCC9ISZ3q5NdqbhdzLZb+yH4hpmXy3I2ioVtGTFE'
                'myYZxbwcdpzvu68eku8+cGmf/GZAdz9IeaeOGMM0ERtx3P30z24+c6d86jqc'
                'uOP8Il18EdE/DP8L3w8fvnegezyl/Glxq/BaPljhTe1l9LOUPoj15YBfbB5n'
                'YoXTqKvZ4dTW2eZxPLZx+j2xlLy6Ni4SgWwpozfmPUj8fuvhuhK/lGUMRxgZ'
                'TmArV9GYVOU4s+qOLunodlGzo3mgeZMcxbwZir9p8QZFpj4C/kHnUfKO+YJ5'
                'NJ7z6OPsYP8rxuV3NcAFuAAX4P43XNqD63c/pLUZUzO43YCEVXBjnPINcOON'
                'S4OgXwPc8DipvO35Ut2/kfZfQO9+ewG6+03K3s04TW9topsa5ahyvYut7Yuv'
                'Wc+Gdj/P52sKz9ptPOXWK5AzuU8tb5ja9TuRbal4Q1rvCNT6zdzA561DWHwW'
                'pnvXXa13Zxx+bw3DFwn9zffYBxdxKidxP8Fg47zYw97NbVj7P/nFW+E=')

        def callback(caller, event):
            self.assertFalse(isinstance(event, cpp.LoadExceptionEvent))
            self.assertFalse(isinstance(event, cpp.RunExceptionEvent))

        T.maybe_download_sbs()
        for windows_mode in ((False, True) if sys.platform.startswith("win")
                             else (False,)):
            pipeline = cpp.Pipeline()
            pipeline.add_listener(callback)
            pipeline.load(StringIO(zlib.decompress(base64.b64decode(data))))
            ipath = os.path.join(T.example_images_directory(), 'ExampleSBSImages')
            bpath = tempfile.mkdtemp()
            bfile = os.path.join(bpath, C.F_BATCH_DATA)
            hfile = os.path.join(bpath, C.F_BATCH_DATA_H5)
            try:
                li = pipeline.modules()[0]
                self.assertTrue(isinstance(li, LI.LoadImages))
                module = pipeline.modules()[1]
                self.assertTrue(isinstance(module, C.CreateBatchFiles))
                li.location.dir_choice = LI.ABSOLUTE_FOLDER_NAME
                li.location.custom_path = ipath
                module.wants_default_output_directory.value = False
                module.custom_output_directory.value = bpath
                module.remote_host_is_windows.value = windows_mode
                self.assertEqual(len(module.mappings), 1)
                mapping = module.mappings[0]
                mapping.local_directory.value = ipath
                self.assertFalse(pipeline.in_batch_mode())
                measurements = cpmeas.Measurements(mode="memory")
                image_set_list = cpi.ImageSetList()
                result = pipeline.prepare_run(
                        cpw.Workspace(pipeline, None, None, None,
                                      measurements, image_set_list))
                self.assertFalse(pipeline.in_batch_mode())
                self.assertFalse(result)
                self.assertFalse(module.batch_mode.value)
                self.assertTrue(measurements.has_feature(
                        cpmeas.EXPERIMENT, cpp.M_PIPELINE))
                pipeline = cpp.Pipeline()
                pipeline.add_listener(callback)
                image_set_list = cpi.ImageSetList()
                measurements = cpmeas.Measurements(mode="memory")
                workspace = cpw.Workspace(pipeline, None, None, None,
                                          cpmeas.Measurements(),
                                          image_set_list)
                workspace.load(hfile, True)
                measurements = workspace.measurements
                self.assertTrue(pipeline.in_batch_mode())
                module = pipeline.modules()[1]
                self.assertTrue(isinstance(module, C.CreateBatchFiles))
                self.assertTrue(module.batch_mode.value)
                image_numbers = measurements.get_image_numbers()
                self.assertTrue([x == i + 1 for i, x in enumerate(image_numbers)])
                pipeline.prepare_run(workspace)
                pipeline.prepare_group(workspace, {}, range(1, 97))
                for i in range(96):
                    image_set = image_set_list.get_image_set(i)
                    for image_name in ('DNA', 'Cytoplasm'):
                        pathname = measurements.get_measurement(
                                cpmeas.IMAGE, "PathName_" + image_name, i + 1)
                        self.assertEqual(pathname,
                                         '\\imaging\\analysis' if windows_mode
                                         else '/imaging/analysis')
                measurements.close()
            finally:
                if os.path.exists(bfile):
                    os.unlink(bfile)
                if os.path.exists(hfile):
                    os.unlink(hfile)
                os.rmdir(bpath)

    def test_04_01_alter_path(self):
        module = C.CreateBatchFiles()
        module.mappings[0].local_directory.value = "foo"
        module.mappings[0].remote_directory.value = "bar"

        self.assertEqual(module.alter_path("foo/bar"), "bar/bar")
        self.assertEqual(module.alter_path("baz/bar"), "baz/bar")

    def test_04_02_alter_path_regexp(self):
        module = C.CreateBatchFiles()
        module.mappings[0].local_directory.value = "foo"
        module.mappings[0].remote_directory.value = "bar"

        self.assertEqual(
                module.alter_path("foo/bar", regexp_substitution=True), "bar/bar")
        self.assertEqual(
                module.alter_path("baz/bar", regexp_substitution=True), "baz/bar")

        module.mappings[0].local_directory.value = r"\foo\baz"
        module.remote_host_is_windows.value = True
        self.assertEqual(
                module.alter_path(r"\\foo\\baz\\bar", regexp_substitution=True),
                r"bar\\bar")

    if sys.platform == 'win32':
        def test_04_03_alter_path_windows(self):
            module = C.CreateBatchFiles()
            module.mappings[0].local_directory.value = "\\foo"
            module.mappings[0].remote_directory.value = "\\bar"

            self.assertEqual(
                    module.alter_path("\\foo\\bar"), "/bar/bar")
            self.assertEqual(
                    module.alter_path("\\FOO\\bar"), "/bar/bar")
            self.assertEqual(
                    module.alter_path("\\baz\\bar"), "/baz/bar")

        def test_04_04_alter_path_windows_regexp(self):
            module = C.CreateBatchFiles()
            module.mappings[0].local_directory.value = "foo"
            module.mappings[0].remote_directory.value = "bar"

            self.assertEqual(
                    module.alter_path("\\\\foo\\\\bar", regexp_substitution=True),
                    "/foo/bar")
            self.assertEqual(
                    module.alter_path("\\\\foo\\g<bar>", regexp_substitution=True),
                    "/foo\\g<bar>")
