from pip._internal import main

# Step 1: Ensure requirements are installed with editable flag
main(["--no-binary", "--editable"])

# If the error persists, manually install 'commons'
pip install commons