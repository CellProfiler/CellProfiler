import click

from ._environment import pass_environment


@pass_environment
def work(context):
    pass
