import argparse
from pathlib import Path

from lazycon import load


def evaluate():
    parser = argparse.ArgumentParser(description='Run the given `config` by evaluating the passed `expression`.')
    parser.add_argument('config', help='path to a config file.')
    parser.add_argument('expression', help='the entry point.')
    # ignoring unknown args as they might be used inside the config
    args = parser.parse_known_args()[0]

    value = load(args.config).eval(args.expression)
    if value is not None:
        print(value)


def build():
    parser = argparse.ArgumentParser(
        description='Read the `input` config and flatten all its imports in order to obtain '
                    'an `output` config without dependencies.')
    parser.add_argument('input', help='path to the input config file.')
    parser.add_argument('output', help='path to the output config file.')
    parser.add_argument('entry_points', nargs='*', help='names that should be kept during rendering.')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    load(args.input).dump(output, args.entry_points or None)
