import hashlib
import json
import os
import re

first_cap = re.compile('(.)([A-Z][a-z]+)')
all_cap = re.compile('([a-z0-9])([A-Z])')


def snake_case(name):
    name = first_cap.sub(r'\1_\2', name)
    return all_cap.sub(r'\1_\2', name).lower()


def walk(path, source, exclude):
    modules = []
    if path in exclude:
        return modules

    for root, dirs, files in os.walk(path):
        for directory in dirs:
            dir_path = os.path.join(root, directory)
            modules.extend(walk(dir_path, '{}.{}'.format(source, directory), exclude))

        for file in files:
            name, extension = os.path.splitext(file)
            if extension == '.py':
                file_path = os.path.join(root, file)
                modules.append((file_path, '{}.{}'.format(source, name)))
        break
    return modules


def handle_corruption(db_path):
    db_path = os.path.realpath(db_path)
    raise RuntimeError('Resources base file corrupted. You may want to delete it: ' + db_path) from None


def read_config(path):
    try:
        with open(path, 'r') as file:
            config = json.load(file)
        try:
            hashes = config['hashes']
            config = config['config']
        except KeyError:
            handle_corruption(path)

    except FileNotFoundError:
        config = []
        hashes = {}

    return config, hashes


def get_hash(path, buffer_size=65536):
    current_hash = hashlib.md5()

    with open(path, 'rb') as file:
        data = file.read(buffer_size)
        while data:
            current_hash.update(data)
            data = file.read(buffer_size)

    return '{}_{}'.format(current_hash.hexdigest(), os.path.getsize(path))
