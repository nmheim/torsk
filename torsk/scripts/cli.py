import click

from torsk.scripts import defaults
from torsk.scripts import scale
from torsk.scripts import animate


@click.group('torsk')
def cli():
    """torsk command line interface"""
    pass


cli.add_command(defaults.cli)
cli.add_command(scale.cli)
cli.add_command(animate.cli)


if __name__ == "__main__":
    cli()