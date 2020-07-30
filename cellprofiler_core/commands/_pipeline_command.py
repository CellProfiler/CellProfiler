import click

from ..__main__ import pass_environment


@click.group("pipeline")
@pass_environment
def command(context):
    pass


@command.command("run")
@click.option("--batch-size", type=click.Path())
@click.option("--data", type=click.Path())
@click.option("--default-input-directory", type=click.Path())
@click.option("--default-output-directory", type=click.Path())
@click.option("--file-list", type=click.Path())
@click.option("--first-image-set", type=click.Path())
@click.option("--last-image-set", type=click.Path())
@click.option("--groups", type=click.Path())
@pass_environment
def run(
    context,
    batch_size,
    data,
    default_input_directory,
    default_output_directory,
    file_list,
    first_image_set,
    last_image_set,
    groups,
):
    pass
