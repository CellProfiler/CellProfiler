import click

from ..__main__ import pass_environment


@click.group("worker")
@pass_environment
def command(context):
    pass


@command.command("start")
@pass_environment
def start(context):
    pass
