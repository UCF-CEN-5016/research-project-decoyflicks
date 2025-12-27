import sys
print(sys.version)

def eval_step():
    passthrough_logs = {'a': 1}
    logs = {'b': 2}
    return passthrough_logs | logs

eval_step()