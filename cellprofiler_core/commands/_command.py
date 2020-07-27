import os

import click

from cellprofiler_core.commands.__main__ import cmd_folder


class Command(click.MultiCommand):
    def list_commands(self, ctx):
        rv = []
        for filename in os.listdir(cmd_folder):
            if filename.endswith(".py") and filename.startswith("cmd_"):
                rv.append(filename[4:-3])
        rv.sort()
        return rv

    def get_command(self, ctx, name):
        try:
            mod = __import__(f"complex.commands.cmd_{name}", None, None, ["cli"])
        except ImportError:
            return
        return mod.main
