from resource_manager import register, register_inline


# the name of the resource will be determined from the class name
@register(module_type='dataset')
class FromIds:
    def __init__(self, ids):
        self.ids = ids

    def load_by_id(self, id):
        x = 1
        # x = load...
        return x

    # some other important methods...


ids = [1, 5, 3, 8, 9, 3]
register_inline(ids, 'ids', 'constants')
