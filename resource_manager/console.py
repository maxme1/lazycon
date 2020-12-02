import argparse

from pathlib import Path

from resource_manager import read_config


def render_config_resource():
    parser = argparse.ArgumentParser(description='Run the given `config` by evaluating the passed `expression`.')
    parser.add_argument('config', help='Path to a config file.')
    parser.add_argument('expression', help='The entry point.')
    # ignoring unknown args as they might be used inside the config
    args = parser.parse_known_args()[0]

    value = read_config(args.config).eval(args.expression)
    if value is not None:
        print(value)


def build_config():
    parser = argparse.ArgumentParser(
        description='Read the `input` config and flatten all its imports in order to obtain '
                    'an `output` config without dependencies.')
    parser.add_argument('input', help='Path to the input config file.')
    parser.add_argument('output', help='Path to the output config file.')
    parser.add_argument('entry_points', nargs='*', help='Names that should be kept during rendering.')
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    read_config(args.input).save_config(output, args.entry_points or None)
