import click

from torsk.scripts import defaults


@click.group('torsk')
def cli():
    """torsk command line interface"""
    pass


cli.add_command(defaults.cli)


if __name__ == "__main__":
    cli()
