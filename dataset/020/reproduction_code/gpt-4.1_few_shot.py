# This minimal example reproduces the TypeError seen in tf-models-official 2.12.0 when run with Python 3.8

dict1 = {'a': 1}
dict2 = {'b': 2}

# The following line causes:
# TypeError: unsupported operand type(s) for |: 'dict' and 'dict'
result = dict1 | dict2

print(result)