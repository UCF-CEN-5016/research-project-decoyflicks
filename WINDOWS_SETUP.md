# Windows Setup Guide for RepGen

This guide provides step-by-step instructions for running RepGen on Windows.

## Prerequisites

- Windows 10/11
- [Git for Windows](https://git-scm.com/download/win) OR [WSL 2](https://learn.microsoft.com/en-us/windows/wsl/install)
- Python 3.12+ (from python.org or Microsoft Store)
- OpenAI API key

## Option 1: Git Bash (Recommended for Most Users)

Git Bash is the simplest solution for Windows users. It's included with Git for Windows.

### Step 1: Install Git for Windows

1. Download [Git for Windows](https://git-scm.com/download/win)
2. Run the installer and accept all default options
3. After installation, you'll have Git Bash available

### Step 2: Install Python

1. Download [Python 3.12+](https://www.python.org/downloads/windows/)
2. **Important:** Check "Add Python to PATH" during installation
3. Verify installation:
   ```bash
   # Open Git Bash and run:
   python --version
   ```

### Step 3: Set Up the Project

1. **Open Git Bash** (Right-click → "Git Bash Here" or search for "Git Bash" in Start Menu)

2. Navigate to the project directory:
   ```bash
   cd /c/path/to/ICSE26-RepGen
   ```
   
   Example for typical Windows paths:
   ```bash
   cd /c/Users/YourUsername/Downloads/Research/ICSE26-RepGen
   ```

3. Create virtual environment:
   ```bash
   python3 -m venv venv
   ```

4. Activate virtual environment:
   ```bash
   source venv/Scripts/activate
   ```

5. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Step 4: Run the Pipeline

1. Set your OpenAI API key:
   ```bash
   export OPENAI_API_KEY="sk-your-key-here"
   ```

2. Run a quick test:
   ```bash
   bash scripts/quick_start.sh 80-82 1
   ```

3. Or use the full pipeline:
   ```bash
   bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --run
   ```

## Option 2: WSL 2 (Windows Subsystem for Linux)

WSL 2 provides a full Linux environment on Windows, which may be more compatible with some tools.

### Step 1: Install WSL 2

1. Open PowerShell as Administrator
2. Run:
   ```powershell
   wsl --install
   ```
3. Restart your computer
4. Open Ubuntu from Start Menu and complete initial setup

### Step 2: Install Python and Git

```bash
sudo apt update
sudo apt install python3.12 python3.12-venv git
```

### Step 3: Set Up the Project

```bash
cd /mnt/c/path/to/ICSE26-RepGen
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Step 4: Run the Pipeline

```bash
export OPENAI_API_KEY="sk-your-key-here"
bash scripts/pipeline.sh --bugs 80-82 --dataset ae_dataset --setup --run
```

## Common Issues and Solutions

### Issue: "bash: command not found"

**Cause:** Git Bash is not installed or not in PATH

**Solution:**
- Install [Git for Windows](https://git-scm.com/download/win)
- Make sure to select Git Bash during installation
- Use Git Bash terminal, not CMD.exe or PowerShell

### Issue: "python: command not found"

**Cause:** Python is not in PATH or not installed

**Solution:**
- Reinstall Python from [python.org](https://www.python.org/downloads/windows/)
- **Make sure to check "Add Python to PATH"**
- Restart Git Bash after installation

### Issue: Virtual environment won't activate

**Solution:** Use the correct activation command for Windows:

```bash
# Git Bash on Windows
source venv/Scripts/activate

# NOT source venv/bin/activate (this is for macOS/Linux)
```

### Issue: OPENAI_API_KEY not working

**Solution:** Make sure to set it in the current Git Bash session:

```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
echo $OPENAI_API_KEY  # Verify it's set
```

If using WSL 2:
```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
# This needs to be set in each new terminal session
```

### Issue: Git clone failing with SSL errors

**Solution:** Disable SSL verification temporarily:

```bash
git config --global http.sslVerify false
```

Or update Git certificates:
1. Download [Git certificate bundle](https://curl.se/docs/caextract.html)
2. Configure Git:
   ```bash
   git config --global http.sslCAinfo /path/to/cacert.pem
   ```

### Issue: Permission denied errors on scripts

**Solution:** Make scripts executable:

```bash
chmod +x scripts/*.sh
```

## File Path Examples

When working with file paths in Windows Git Bash, use forward slashes `/`, not backslashes:

```bash
# Correct (Git Bash)
cd /c/Users/YourUsername/Downloads/Research/ICSE26-RepGen

# Incorrect (Windows paths)
cd C:\Users\YourUsername\Downloads\Research\ICSE26-RepGen
```

Windows path mapping in Git Bash:
- `C:\` → `/c/`
- `D:\` → `/d/`
- `E:\` → `/e/`
- etc.

## Running Commands Across Platforms

All scripts automatically detect your platform and use appropriate commands:

```bash
# Same command works on macOS, Linux, and Windows (Git Bash)
bash scripts/pipeline.sh --bugs 1-10 --setup --run
```

The scripts handle platform-specific differences internally:
- ANSI colors (disabled on Windows CMD, enabled on Git Bash)
- Path separators (handled by Git Bash)
- Virtual environment activation (auto-detected)
- Line endings (automatic with Git Bash)

## Tips and Best Practices

1. **Always use Git Bash on Windows**, not CMD.exe or PowerShell
2. **Keep paths simple**: Avoid special characters and spaces if possible
3. **Use forward slashes** in paths: `/c/path/to/file`, not `C:\path\to\file`
4. **One terminal per task**: Open a new Git Bash window for different bug runs
5. **Check API key regularly**: Re-export if terminal restarts
6. **Monitor disk space**: Repository clones are cached; clean `.code_cache/` if space is low

## Additional Resources

- [Git Bash Documentation](https://git-scm.com/docs)
- [WSL 2 Guide](https://learn.microsoft.com/en-us/windows/wsl/)
- [Python on Windows](https://docs.python.org/3/using/windows.html)
- [OpenAI API Keys](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)

## Next Steps

After setting up, see [PIPELINE.md](PIPELINE.md) for:
- Complete usage guide
- All available options
- Example workflows
- Troubleshooting
