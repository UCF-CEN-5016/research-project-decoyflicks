# Windows Setup Guide for RepGen

This guide provides step-by-step instructions for running RepGen on Windows using Git Bash or WSL 2.

**First time?** Start with [README.md](README.md) for prerequisites overview, then follow the relevant option below.

---

## Option 1: Git Bash (Recommended for Most Users)

Git Bash is the simplest solution. It's included with Git for Windows and provides a full bash shell on Windows.

### Step 1: Install Git for Windows

1. Download [Git for Windows](https://git-scm.com/download/win)
2. Run the installer with default options
3. Verify by opening a new terminal and running:
   ```bash
   git --version
   ```

### Step 2: Verify Python Installation

1. Download [Python 3.12+](https://www.python.org/downloads/windows/) if not already installed
2. **Important:** During installation, check "Add Python to PATH"
3. Verify in Git Bash:
   ```bash
   python --version
   ```

### Step 3: Set Up the Project

1. **Open Git Bash** (Right-click → "Git Bash Here" or search "Git Bash" in Start Menu)

2. Navigate to your project directory:
   ```bash
   cd /c/path/to/ICSE26-RepGen
   ```
   
   Example for typical Windows paths:
   ```bash
   cd /c/Users/YourUsername/Downloads/Research/ICSE26-RepGen
   ```

3. Create virtual environment:
   ```bash
   python -m venv venv
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

---

## Option 2: WSL 2 (Windows Subsystem for Linux)

WSL 2 provides a full Linux environment on Windows for advanced users.

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

---

## Comparison: Git Bash vs WSL 2

| Aspect | Git Bash | WSL 2 |
|--------|----------|-------|
| **Setup** | Easy (1 click) | Moderate (needs restart) |
| **Speed** | Good | Excellent |
| **Linux compatibility** | Partial | Full |
| **File access** | Native | Via `/mnt/` |
| **For beginners** | ✅ Recommended | Advanced users |

---

## Common Issues and Solutions

### Issue: "bash: command not found"

**Cause:** Git Bash is not installed or not in PATH

**Solution:**
1. Install [Git for Windows](https://git-scm.com/download/win)
2. Make sure to accept Git Bash installation
3. Use Git Bash terminal, not CMD.exe or PowerShell

### Issue: "python: command not found"

**Cause:** Python is not in PATH or not installed

**Solution:**
1. Download [Python 3.12+](https://www.python.org/downloads/windows/)
2. **Critically important:** Check "Add Python to PATH" during installation
3. Restart Git Bash after installation
4. Verify: `python --version`

### Issue: "python: command not found" in WSL 2

**Solution:**
```bash
sudo apt update
sudo apt install python3.12 python3.12-venv
```

### Issue: Virtual environment won't activate

**Solution:** Use the correct activation command for your setup:

**Git Bash on Windows:**
```bash
source venv/Scripts/activate
```

**WSL 2 or Linux:**
```bash
source venv/bin/activate
```

**Wrong command (don't use on Windows Git Bash):**
```bash
source venv/bin/activate  # This won't work on Windows Git Bash
```

### Issue: OPENAI_API_KEY not working

**Cause:** API key not set in current terminal session

**Solution:** Set it in the current Git Bash/WSL session:
```bash
export OPENAI_API_KEY="sk-your-actual-key-here"
echo $OPENAI_API_KEY  # Verify it's set
```

To make it persistent across sessions, add to your shell profile:
- **Git Bash:** `~/.bashrc`
- **WSL 2:** `~/.bashrc`

Edit with:
```bash
nano ~/.bashrc
# Add: export OPENAI_API_KEY="sk-..."
# Save: Ctrl+O, Enter, Ctrl+X
```

### Issue: Git clone failing with SSL errors

**Solution:** Update Git or temporarily disable SSL:

```bash
# Try updating Git first
git --version

# If still failing, try:
git config --global http.sslVerify false
```

### Issue: Permission denied on scripts

**Solution:** Make scripts executable:

```bash
chmod +x scripts/*.sh
```

---

## File Path Conversion

Windows paths need to be converted for Git Bash (use forward slashes):

| Windows Path | Git Bash Path |
|--------------|---------------|
| `C:\Users\...` | `/c/Users/...` |
| `D:\Projects\...` | `/d/Projects/...` |
| `E:\...` | `/e/...` |

### Examples

```bash
# Navigate to typical Windows locations
cd /c/Users/YourUsername/Downloads/Research/ICSE26-RepGen
cd /d/Projects/ICSE26-RepGen

# Always use forward slashes in Git Bash
cd /c/path/to/file  # ✅ Correct
cd C:\path\to\file  # ❌ Wrong
```

---

## Terminal Tips

### Always Use Git Bash on Windows

| Terminal | Works? | Recommendation |
|----------|--------|-----------------|
| Git Bash | ✅ Yes | **Use this** |
| CMD.exe | ❌ No | Don't use |
| PowerShell | ❌ No | Don't use |
| WSL 2 | ✅ Yes | Use if already set up |

### Opening Git Bash

1. **Method 1:** Right-click in folder → "Git Bash Here"
2. **Method 2:** Search "Git Bash" in Start Menu
3. **Method 3:** Open from Program Files

---

## Next Steps

1. Follow either **Option 1 (Git Bash)** or **Option 2 (WSL 2)** above
2. Once setup is complete, see [README.md](README.md) for quick start
3. For detailed pipeline usage, see [OPENAI_PIPELINE.md](OPENAI_PIPELINE.md) or [QWEN_PIPELINE.md](QWEN_PIPELINE.md)

---

## Troubleshooting by Error Message

| Error | Cause | Solution |
|-------|-------|----------|
| `bash: command not found` | Git Bash not installed | Install Git for Windows |
| `python: command not found` | Python not in PATH | Reinstall Python, check "Add to PATH" |
| `OPENAI_API_KEY not set` | API key not exported | `export OPENAI_API_KEY="sk-..."` |
| `venv: command not found` | Python venv not available | Use `python -m venv venv` |
| `Permission denied` | Scripts not executable | `chmod +x scripts/*.sh` |
| `SSL certificate error` | Git SSL issue | `git config --global http.sslVerify false` |

---

## Additional Resources

- [Git Bash Documentation](https://git-scm.com/docs)
- [WSL 2 Guide](https://learn.microsoft.com/en-us/windows/wsl/)
- [Python on Windows](https://docs.python.org/3/using/windows.html)
- [OpenAI API Keys](https://help.openai.com/en/articles/4936850-where-do-i-find-my-secret-api-key)

---

**Ready?** Start with either Git Bash or WSL 2 option above!
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

After setting up, see [OPENAI_PIPELINE.md](OPENAI_PIPELINE.md) or [QWEN_PIPELINE.md](QWEN_PIPELINE.md) for:
- Complete usage guide
- All available options
- Example workflows
- Troubleshooting
