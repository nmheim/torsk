import click

from torsk.scripts import defaults
from torsk.scripts import scale
from torsk.scripts import animate
from torsk.scripts import ncextract
from torsk.scripts import analyse
from torsk.scripts import detect
from torsk.scripts import anomaly_count


@click.group('torsk')
def cli():
    """torsk command line interface"""
    pass


cli.add_command(defaults.cli)
cli.add_command(scale.cli)
cli.add_command(animate.cli)
cli.add_command(ncextract.cli)
cli.add_command(analyse.cli)
cli.add_command(detect.cli)
cli.add_command(anomaly_count.cli)


if __name__ == "__main__":
    cli()
