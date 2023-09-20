import click

from ..__main__ import pass_environment


@click.group("pipeline")
@pass_environment
def command(context):
    pass


@command.command("measurements", help="returns measurements extracted by the pipeline")
@pass_environment
def measurements(context):
    pass


@command.command("run", help="executes the pipeline")
@click.argument("pipeline", type=click.File("r"))
@click.option("--batch-size", type=int)
@click.option("--data", type=click.Path())
@click.option("--default-images-directory", type=click.Path())
@click.option("--default-output-directory", type=click.Path())
@click.option("--images", multiple=True, type=click.Path())
@click.option("--beginning", default=1, type=int)
@click.option("--end", type=int)
@click.option("--group", type=str)
@pass_environment
def run(
    context,
    batch_size,
    data,
    images_directory,
    output_directory,
    images,
    beginning,
    end,
    grouping,
):
    pass
