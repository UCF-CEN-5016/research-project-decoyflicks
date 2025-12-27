import os

from dominoai import domino as cd

# Initialize database connection or set environment variables as needed
os.environ['DB_URI'] = 'your-database-connection-string'

# Trigger any setup required for cdb before using it
try:
    cd.init()  # Replace with actual initialization method if available
except Exception as e:
    print(f"Initialization failed: {e}")

# Now call the problematic method to trigger the error
try:
    tensor = [[1, 2], [3, 4]]
    result = cd.all_reduce(tensor)
except AttributeError as e:
    print(f"AttributeError occurred: {e}")