import argparse

from autocommand import autocommand

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
    @autocommand(True)
    def _build_config(source, target):
        """Expand all config imports in `source` to create a single config with all resources and put it to `target`."""
        read_config(source).save_config(target)
