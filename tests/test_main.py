import os.path
import re

import pytest

import cellprofiler.__main__


@pytest.fixture(scope="module")
def resources():
        return os.path.join(os.path.dirname(__file__), "resources", "test_main")


def test_get_batch_commands_grouped_batch_data(resources, capsys):
    batch_file = os.path.join(resources, "grouped_batch_data.h5")

    cellprofiler.__main__.get_batch_commands(batch_file)

    batch_commands = capsys.readouterr()[0].strip().split("\n")

    assert len(batch_commands) == 3

    expected_groups = [(u"1", u"3"), (u"4", u"6"), (u"7", u"9")]

    groups = [re.match(".* -f ([0-9]) -l ([0-9])", batch_command).groups() for batch_command in batch_commands]

    assert groups == expected_groups


def test_get_batch_commands(resources, capsys):
    batch_file = os.path.join(resources, "batch_data.h5")

    cellprofiler.__main__.get_batch_commands(batch_file)

    batch_commands = capsys.readouterr()[0].strip().split("\n")

    assert len(batch_commands) == 9

    for idx, batch_command in enumerate(batch_commands):
        first, last = re.match(".* -f ([0-9]) -l ([0-9])", batch_command).groups()

        assert first == str(idx + 1)

        assert last == first


def test_get_batch_commands_grouped_by_metadata_batch_data(resources, capsys):
    batch_file = os.path.join(resources, "metadata_batch_data.h5")

    cellprofiler.__main__.get_batch_commands(batch_file)

    batch_commands = capsys.readouterr()[0].strip().split("\n")

    assert len(batch_commands) == 3

    expected_groups = [u"g01", u"g02", u"g03"]

    groups = [re.match(".* -g Metadata_Plate=(g0[0-9])", batch_command).groups()[0] for batch_command in batch_commands]

    assert groups == expected_groups
