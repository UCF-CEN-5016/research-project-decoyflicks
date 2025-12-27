dict1 = {'a': 1}
dict2 = {'b': 2}
merged = dict1 | dict2
print(merged)

# This code will raise TypeError in Python 3.8
dict1 = {'a': 1, 'b': 2}
dict2 = {'c': 3, 'd': 4}
merged_dict = dict1 | dict2  # This line causes the error
print("Merged dictionary:", merged_dict)