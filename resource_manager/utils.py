from collections import defaultdict


def reverse_mapping(mapping: dict) -> dict:
    result = defaultdict(list)
    for name, node in mapping.items():
        result[node].append(name)
    return dict(result)
