import os
import pathlib

import click

import cellprofiler_core.commands

CONTEXT_SETTINGS = dict(auto_envvar_prefix="COMPLEX")


class Command(click.MultiCommand):
    def get_command(self, context, name):
        try:
            name = f"cellprofiler_core.commands._{name}_command"

            imported_module = __import__(name, None, None, ["command"])
        except ImportError:
            return

        return imported_module.command

    def list_commands(self, context):
        command_names = []

        commands_pathname = cellprofiler_core.commands.__file__

        commands_directory = pathlib.Path(commands_pathname).parent

        for filename in os.listdir(commands_directory):
            if filename.endswith("_command.py") and filename.startswith("_"):
                command_name = filename[1:-11]

                command_names += [command_name]

        command_names.sort()

        return command_names


class Environment:
    def __init__(self):
        pass


pass_environment = click.make_pass_decorator(Environment, ensure=True)


@click.command(cls=Command, context_settings=CONTEXT_SETTINGS)
@pass_environment
def main(context):
    pass


if __name__ == "__main__":
    main({})
