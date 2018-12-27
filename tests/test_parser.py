import os
import unittest

from resource_manager.parser import parse_string


def standardize(source):
    parents, imports, definitions = parse_string(source)
    result = '\n'.join(imp.to_str() for imp in parents) + '\n'
    result += '\n'.join(imp.to_str([name]) for name, imp in imports) + '\n'
    for name, definition in definitions:
        result += definition.to_str([name])
    return result


class TestParser(unittest.TestCase):
    def test_idempotency(self):
        folder = os.path.dirname(__file__)

        for root, _, files in os.walk(folder):
            for filename in files:
                # if filename != 'target.config':
                #     continue

                if filename.endswith('.config'):
                    path = os.path.join(root, filename)
                    with self.subTest(filename=filename):
                        with open(path, 'r') as file:
                            source = file.read()
                        temp = standardize(source)
                        self.assertEqual(temp, standardize(temp))

    def test_unexpected_token(self):
        with self.assertRaises(SyntaxError):
            parse_string('a = [1, 2 3]')

    def test_unexpected_eof(self):
        with self.assertRaises(SyntaxError):
            parse_string('a = [1, 2')

    def test_unrecognized_token(self):
        with self.assertRaises(SyntaxError):
            parse_string('$')
