import unittest
import os

from resource_manager.parser import Parser
from resource_manager.tokenizer import tokenize


def standardize(source):
    tokens = tokenize(source, 4)
    result = ''
    definitions, parents = Parser(tokens).parse()
    if parents:
        result += '@extends ' + ' '.join(f'"{x}"' for x in parents)
    for definition in definitions:
        result += definition.to_str(0) + '\n'
    return result


class TestParser(unittest.TestCase):
    def test_idempotency(self):
        folder = os.path.join(os.path.dirname(__file__), 'idempotency')

        for i, filename in enumerate(os.listdir(folder)):
            path = os.path.join(folder, filename)
            with self.subTest(filename=filename):
                with open(path, 'r') as file:
                    source = file.read()
                temp = standardize(source)
                self.assertEqual(temp, standardize(temp))

# TODO: add exceptions testing
