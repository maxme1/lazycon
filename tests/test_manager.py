import unittest

import numpy as np

from resource_manager.resource_manager import ResourceManager


def mock_get_resource(module_type, module_name):
    name = f'{module_type}.{module_name}'

    def x(**kwargs):
        return name

    return x


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
        try:
            rm = ResourceManager('extends/imports.config')
            self.assertEqual(rm.x, 3)
        except RecursionError:
            self.fail()

    def test_extends(self):
        rm = ResourceManager('extends/inheritance.config', get_module=mock_get_resource)
        for resource in ['just_another_resource', 'one', 'dataset', 'one', 'two', 'three', 'four', 'deeper']:
            with self.subTest(i=resource):
                try:
                    getattr(rm, resource)
                except AttributeError:
                    self.fail()

    def test_from(self):
        rm = ResourceManager('extends/from.config', get_module=mock_get_resource)
        for resource in ['just_another_resource', 'one', 'dataset', 'one', 'two', 'three', 'four', 'deeper']:
            with self.subTest(i=resource):
                try:
                    getattr(rm, resource)
                except AttributeError:
                    self.fail()

    def test_cycle_import(self):
        try:
            ResourceManager('extends/cycle_import.config')
        except RecursionError:
            self.fail()

    def test_inheritance_order(self):
        rm = ResourceManager('extends/order1.config')
        self.assertEqual(rm.one, 1)
        self.assertEqual(rm.two, '2')
        rm = ResourceManager('extends/order2.config')
        self.assertIsNone(rm.one)
        self.assertIsNone(rm.two)

    def test_attr_error(self):
        rm = ResourceManager('extends/inheritance.config')
        with self.assertRaises(AttributeError):
            rm.undefined_value

    def test_functions(self):
        rm = ResourceManager('idempotency/function.config')
        np.testing.assert_array_equal(rm.mean, [2, 5])
        np.testing.assert_array_equal(rm.mean2, [2, 5])
        np.testing.assert_array_almost_equal(rm.std(), [0.81, 0.81], decimal=2)
        self.assertEqual(rm.random.shape, (1, 1, 2, 2))

    def test_build_config(self):
        rm = ResourceManager('extends/inheritance.config')
        with open('extends/built') as built:
            self.assertEqual(rm._get_whole_config(), built.read())

    def test_cycles(self):
        with self.assertRaises(RuntimeError):
            ResourceManager('misc/cycles.config')

    def test_undefined(self):
        with self.assertRaises(RuntimeError):
            ResourceManager('misc/static_undefined.config')

    def test_duplicates(self):
        with self.assertRaises(SyntaxError):
            ResourceManager('misc/duplicate.config')

    def test_exc_handling(self):
        rm = ResourceManager('idempotency/module.config', get_module=raises)
        with self.assertRaises(RuntimeError):
            rm.dataset
