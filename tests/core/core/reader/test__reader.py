from cellprofiler_core.reader import fill_readers, filter_active_readers
from cellprofiler_core.constants.reader import ALL_READERS, AVAILABLE_READERS


def init_readers():
    fill_readers()
    print(ALL_READERS)
    print("---")
    filter_active_readers([x for x in ALL_READERS.keys() if "Google Cloud Storage" not in x], by_module_name=False)
    print(AVAILABLE_READERS)

class TestReaders:
    def test_readers(self):
        init_readers()
