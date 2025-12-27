# Minimal reproduction of the bug in Python 3.8
passthrough_logs = {"a": 1}
logs = {"b": 2}

# This will raise TypeError in Python 3.8
result = passthrough_logs | logs
print(result)

merged_logs = passthrough_logs.copy()
merged_logs.update(logs)
print(merged_logs)

merged_logs = {**passthrough_logs, **logs}
print(merged_logs)