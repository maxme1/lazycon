import unittest

from resource_manager.resource_manager import ResourceManager


def mock_get_resource(module_type, module_name):
    name = f'{module_type}.{module_name}'

    def x(**kwargs):
        return name

    return x


class TestResourceManager(unittest.TestCase):
    def test_import(self):
        rm = ResourceManager('extends/inheritance', mock_get_resource)
        for resource in ['just_another_resource', 'one', 'dataset', 'one', 'two', 'three', 'four', 'deeper']:
            with self.subTest(i=resource):
                try:
                    getattr(rm, resource)
                except AttributeError:
                    self.fail()

    def test_cycle_import(self):
        try:
            ResourceManager('extends/cycle_import', mock_get_resource)
        except RecursionError:
            self.fail()

    def test_inheritance_order(self):
        rm = ResourceManager('extends/order1', mock_get_resource)
        self.assertIsNone(rm.one)
        self.assertIsNone(rm.two)
        rm = ResourceManager('extends/order2', mock_get_resource)
        self.assertEqual(rm.one, 1)
        self.assertEqual(rm.two, '2')

    def test_attr_error(self):
        rm = ResourceManager('extends/inheritance', mock_get_resource)
        with self.assertRaises(AttributeError):
            rm.undefined_value

    def test_build_config(self):
        rm = ResourceManager('extends/inheritance', mock_get_resource)
        print(rm._get_whole_config())
        with open('extends/built') as built:
            self.assertEqual(rm._get_whole_config(), built.read())
