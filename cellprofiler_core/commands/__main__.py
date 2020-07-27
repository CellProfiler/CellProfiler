import os

import click

from ._command import Command
from ._environment import pass_environment

CONTEXT_SETTINGS = dict(auto_envvar_prefix="COMPLEX")

cmd_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "commands"))


@click.command(cls=Command, context_settings=CONTEXT_SETTINGS)
@pass_environment
def main(context):
    pass
