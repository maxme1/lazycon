import argparse
from pathlib import Path

from lazycon import load


def run(config, expression):
    value = load(config).eval(expression)
    if value is not None:
        print(value)


def build(input, output, entry_points):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    load(input).dump(output, entry_points)


def main():
    parser = argparse.ArgumentParser(description='Shortcuts for operations with config files.')
    subparsers = parser.add_subparsers()

    new = subparsers.add_parser('run', help='Run the given `config` by evaluating the passed `expression`.')
    new.set_defaults(callback=run)
    new.add_argument('config', help='path to a config file.')
    new.add_argument('expression', help='the entry point.')

    new = subparsers.add_parser(
        'build', help='Read the `input` config and flatten all its imports in order to obtain '
                      'an `output` config without dependencies.')
    new.set_defaults(callback=build)
    new.add_argument('input', help='path to the input config file.')
    new.add_argument('output', help='path to the output config file.')
    new.add_argument('-ep', '--entry_points', '--entrypoints', '--entrypoint', nargs='+', default=None,
                     help='names that should be kept during rendering.')

    args = vars(parser.parse_args())
    if 'callback' not in args:
        parser.print_help()
    else:
        callback = args.pop('callback')
        callback(**args)
