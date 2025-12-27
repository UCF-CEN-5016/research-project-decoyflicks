# Improved implementation in Python 3.8
passthrough_logs = {"a": 1}
logs = {"b": 2}

# Merging dictionaries using the update method
merged_logs = passthrough_logs.copy()
merged_logs.update(logs)
print(merged_logs)

# Merging dictionaries using the **kwargs syntax
merged_logs = {**passthrough_logs, **logs}
print(merged_logs)