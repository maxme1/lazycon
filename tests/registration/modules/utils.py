from resource_manager import register


# the resource name will be determined from the function's name
@register(module_type='utils')
def unique_ids(dataset):
    return list(set(dataset.ids))


@register(module_type='utils')
def load_from_dataset(id, dataset):
    # some logic. maybe loading from disk
    return "probably some numpy array"
