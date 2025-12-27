# Improved code:

# Merge dictionaries using the update() method
dict1 = {'a': 1}
dict2 = {'b': 2}
merged_dict = dict(dict1)
merged_dict.update(dict2)
print("Merged dictionary:", merged_dict)

# Merge dictionaries using dictionary unpacking (**)
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
merged_dict = {**dict1, **dict2}
print("Merged dictionary:", merged_dict)