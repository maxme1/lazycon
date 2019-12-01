import argparse

from resource_manager import read_config


def render_config_resource():
    parser = argparse.ArgumentParser(description='Run the given `config` using `name` as an entry point.')
    parser.add_argument('config', help='Path to a config file.')
    parser.add_argument('name', help='The entry point.')
    # ignoring unknown args as they might be used inside the config
    args = parser.parse_known_args()[0]

    value = read_config(args.config).get_resource(args.name)
    if value is not None:
        print(value)


def build_config():
    parser = argparse.ArgumentParser(
        description='Read the `input` config and flatten all its imports in order to obtain '
                    'an `output` config without dependencies.')
    parser.add_argument('input', help='Path to the input config file.')
    parser.add_argument('output', help='Path to the output config file.')
    args = parser.parse_args()
    read_config(args.input).save_config(args.output)
