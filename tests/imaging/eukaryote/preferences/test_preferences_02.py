class TestPreferences_02(unittest.TestCase):
    def setUp(self):
        # force the test to use a special config object
        class FakeConfig:
            def Exists(self, arg):
                return True
            def Read(self, arg):
                return None
            def Write(self, arg, val):
                pass
            def GetEntryType(self, kwd):
                return 1
        self.old_headless = cpprefs.__dict__['__is_headless']
        self.old_headless_config = cpprefs.__dict__['__headless_config']
        cpprefs.__dict__['__is_headless'] = True
        cpprefs.__dict__['__headless_config'] = FakeConfig()
        
    def tearDown(self):
        cpprefs.__dict__['__is_headless'] = self.old_headless
        cpprefs.__dict__['__headless_config'] = self.old_headless_config

    def test_01_01_default_directory_none(self):
        print cpprefs.get_default_image_directory()
        self.assertTrue(True);
