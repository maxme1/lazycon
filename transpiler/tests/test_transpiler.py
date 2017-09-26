import unittest
import json
import os

from transpiler import transpile


def compare_json(a, b):
    if type(a) != type(b):
        return False
    if type(a) is dict:
        if set(a.keys()) ^ set(b.keys()):
            return False
        for key in a.keys():
            if not compare_json(a[key], b[key]):
                return False
        return True
    if type(a) is list:
        if len(a) != len(b):
            return False
        for i, j in zip(a, b):
            if not compare_json(i, j):
                return False
        return True
    return a == b


class TestTranspiler(unittest.TestCase):
    def test_equal(self):
        folder = os.path.join(os.path.dirname(__file__), 'compare')

        for i, filename in enumerate(os.listdir(folder)):
            path = os.path.join(folder, filename)
            with self.subTest(i=filename):
                with open(path, 'r') as file:
                    data = file.read().split('\n:becomes:\n')
                a = json.loads(transpile(data[0]))
                b = json.loads(data[1])
                self.assertTrue(compare_json(a, b))


# TODO: add exceptions testing

if __name__ == '__main__':
    unittest.main()
