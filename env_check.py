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

    print("\n🎯 Pre-check complete.")

    if success:
        sys.exit(0)  # success
    else:
        sys.exit(1)  # failure

if __name__ == "__main__":
    main()