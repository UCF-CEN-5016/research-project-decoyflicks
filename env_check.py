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

    print("\n🎯 Pre-check complete.")

    if success:
        sys.exit(0)  # success
    else:
        sys.exit(1)  # failure

    print("\n🎯 Pre-check complete.")

if __name__ == "__main__":
    main()