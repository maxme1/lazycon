from resource_manager import register, register_inline


# the name of the resource will be determined from the class name
@register(module_type='dataset')
class FromIds:
    def __init__(self, ids):
        self.ids = ids

        # some other important methods...
