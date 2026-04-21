import sys
import shutil
import os

def check_python_version():
    print("Checking Python version...")
    version = sys.version_info

    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version OK: {version.major}.{version.minor}")
    else:
        print(f"❌ Python version too low: {version.major}.{version.minor}")
        print("👉 Please install Python 3.8 or higher\n")

def check_disk_space():
    print("\nChecking disk space...")
    
    total, used, free = shutil.disk_usage("/")

    free_gb = free // (2**30)

    if free_gb >= 5:
        print(f"✅ sufficient disk space: {free_gb} GB available")
    else:
        print(f"❌ Not enough disk space: {free_gb} GB available")
        print("👉 At least 5GB required for LLM models\n")

def check_dependencies():
    print("\nChecking Python dependencies...")

    required_packages = ["numpy", "torch"]

    missing = []

    for pkg in required_packages:
        try:
            __import__(pkg)
            print(f"✅ {pkg} is installed")
        except ImportError:
            print(f"❌ {pkg} is NOT installed")
            missing.append(pkg)

    return missing

def check_llm_configuration():
    print("\nChecking LLM configuration...")

    local_model_path = "./models"   # adjust if your repo uses a different path
    api_key = os.getenv("OPENAI_API_KEY")

    local_exists = os.path.exists(local_model_path) and len(os.listdir(local_model_path)) > 0
    api_exists = api_key is not None and api_key != ""

    if local_exists:
        print("✅ Local LLM detected (models directory found)")
    
    if api_exists:
        print("✅ API key detected (OPENAI_API_KEY is set)")

    if not local_exists and not api_exists:
        print("❌ No LLM configuration found")
        print("👉 Option 1: Add models to ./models directory")
        print("👉 Option 2: Set OPENAI_API_KEY environment variable")
        return False

    return True

def check_required_files():
    print("\nChecking required files and paths...")

    required_files = [
        "RepGen.sh",
        "README.md"
    ]

    missing = []

    for file in required_files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
            missing.append(file)

    if missing:
        print("\n👉 Missing required files:", ", ".join(missing))
        return False

    return True

def main():
    print("🔍 Running Environment Pre-Check...\n")

    success = True

    # Python version
    print("Checking Python version...")
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python version OK: {version.major}.{version.minor}")
    else:
        print(f"❌ Python version too low: {version.major}.{version.minor}")
        success = False

    # Disk space
    print("\nChecking disk space...")
    total, used, free = shutil.disk_usage("/")
    free_gb = free // (2**30)

    if free_gb >= 5:
        print(f"✅ sufficient disk space: {free_gb} GB available")
    else:
        print(f"❌ Not enough disk space: {free_gb} GB available")
        success = False

    # Dependencies
    missing = check_dependencies()
    if missing:
        print("\n👉 Missing packages:", ", ".join(missing))
        print("👉 Install using: pip install " + " ".join(missing))
        success = False
    
    # LLM check
    if not check_llm_configuration():
        success = False

    # Required files
    if not check_required_files():
        success = False

    print("\n🎯 Pre-check complete.")

    if success:
        sys.exit(0)  # success
    else:
        sys.exit(1)  # failure

if __name__ == "__main__":
    main()