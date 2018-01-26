import unittest

import numpy as np

from resource_manager.resource_manager import ResourceManager, read_config


# TODO: replace with mock
def raises(module_type, module_name):
    name = f'{module_type}.{module_name}'

    if name == 'dataset.isles':
        raise KeyError

    def x(**kwargs):
        return name

    return x


class TestResourceManager(unittest.TestCase):
    def test_import(self):
        rm = read_config('imports/imports.config')
        self.assertEqual(rm.numpy, np)

    def test_cycle_import(self):
        try:
            read_config('imports/cycle_import.config')
        except RecursionError:
            self.fail()

    def test_inheritance_order(self):
        rm = read_config('imports/order1.config')
        self.assertEqual(rm.one, 1)
        self.assertEqual(rm.two, '2')
        rm = read_config('imports/order2.config')
        self.assertIsNone(rm.one)
        self.assertIsNone(rm.two)

    def test_attr_error(self):
        rm = read_config('imports/imports.config')
        with self.assertRaises(AttributeError):
            rm.undefined_value

    def test_items(self):
        rm = read_config('expressions/tail.config')
        self.assertEqual(rm.part, 6)
        np.testing.assert_array_equal(rm.another_part, [1, 2, 3])

    def test_functions(self):
        rm = read_config('expressions/tail.config')
        np.testing.assert_array_equal(rm.mean, [2, 5])
        np.testing.assert_array_equal(rm.mean2, [2, 5])
        np.testing.assert_array_almost_equal(rm.std(), [0.81, 0.81], decimal=2)
        self.assertEqual(rm.random.shape, (1, 1, 2, 2))

    def test_build_config(self):
        rm = read_config('imports/config_import.config', shortcuts={'expressions': 'expressions'})
        with open('imports/built.config') as built:
            self.assertEqual(rm.render_config(), built.read())

    def test_cycles(self):
        with self.assertRaises(RuntimeError):
            read_config('misc/cycles.config')

    def test_undefined(self):
        with self.assertRaises(RuntimeError):
            read_config('misc/static_undefined.config')

    def test_duplicates(self):
        with self.assertRaises(SyntaxError):
            read_config('misc/duplicate.config')

    def test_exc_handling(self):
        rm = read_config('expressions/module.config', get_module=raises)
        with self.assertRaises(RuntimeError):
            rm.dataset
