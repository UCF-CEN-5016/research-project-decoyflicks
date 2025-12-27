# File: infer.py
# Simulate the error by trying to import a local module named 'commons'

try:
    import commons
except ModuleNotFoundError as e:
    print(f"Caught error: {e}")

# Expected: ImportError because 'commons' module is not found in PYTHONPATH or current directory