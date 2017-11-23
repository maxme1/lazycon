import unittest

import functools

from resource_manager import ResourceManager, generate_config, get_module


def get_rm(config_path):
    db_path = 'registration/database.json'
    generate_config('registration', db_path, 'registration')
    get_module_ = functools.partial(get_module, db_path=db_path)
    return ResourceManager(config_path, get_module_)


class TestRegistration(unittest.TestCase):
    def test_registration(self):
        try:
            rm = get_rm('registration/base.config')
            # `ids` is a resource defined in the config
            for id_ in rm.ids:
                # and `load` too
                x = rm.load(id_)
                # do some stuff with x...
            self.assertEqual(1, rm.loader(rm.ids[0]))
            self.assertEqual(rm.aggregated, 3)
            self.assertIsNotNone(rm.rand_number)

        except Exception:
            self.fail()
