import click


class Environment:
    def __init__(self):
        pass


pass_environment = click.make_pass_decorator(Environment, ensure=True)
