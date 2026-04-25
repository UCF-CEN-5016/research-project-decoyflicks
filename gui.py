from tkinter import *
from tkinter import ttk
from tkinter import scrolledtext as st
import subprocess
import threading

# configure root window
root = Tk()
root.title("Imitation Game")
root.configure(bg = "white")
root.minsize(300, 300)
root.geometry("850x450")

# tk vars
bug_start = StringVar()
bug_end = StringVar()
progress = IntVar()
progress.set(0)

# global vars
script = ""
executing = False
thread = ""
run_output = ""
bug_count = 0

# set progress to correct integer value between 0 and 100 = percent complete
def set_progress(completed, total):
    progress.set(int((completed / total) * 100))

# return correct bash string from script for passed bug range; return empty string if script invalid
def get_bash(start, end):
    if script == "cloud.sh" or script == "local.sh":
        return f"bash scripts/pipeline/{script} --quiet --run --bugs {start}-{end}"
    elif script == "baseline.sh" or script == "ablations.sh":
        return f"bash scripts/experimental/{script} --quiet --bugs {start}-{end}"
    else:
        return ""

# check for results every 10 seconds
def update(start, end, done = 0):
    global run_output
    if run_output != "":
        # if run finished, process output
        temp = str(run_output).replace("b\'", "\'").replace("\'", "")
        run_output = eval(repr(temp).replace("\\\\", "\\")).replace("\xe2\x9c\x93", "\t")
        run_output = f"=== Bug {start} ===\n" + run_output + "\n\n"

        # update progress indicators
        set_progress(done + 1, bug_count)
        ratio.config(text = f"\tComplete: {done + 1}/{bug_count} ({progress.get()}%)")

        # display results
        results.configure(state = "normal")
        results.insert(INSERT, run_output)
        results.configure(state = "disabled")
        run_output = ""

        # complete method, no further iterations
        if start == end:
            global executing
            executing = False
        else:
            # run next bug
            global thread
            thread = threading.Thread(target = run, args = (start + 1,), daemon = True)
            thread.start()

            root.after(10000, update, start + 1, end, done + 1)

    else:
        # if no results yet, wait another 10 seconds
        root.after(10000, update, start, end, done)

# run the specified script for a single bug; this should be run on a separate thread
def run(bug):
    print(f"Bug {bug} started!")
    bash_str = get_bash(bug, bug)
    global run_output
    run_output = subprocess.check_output(bash_str, shell = True, stderr = subprocess.STDOUT)
    print(f"Bug {bug} complete!")

def execute():
    # do nothing if already executing
    global executing
    if executing == True:
        return

    # confirm specified bugs are integers
    if not bug_start.get().isdigit() or not bug_end.get().isdigit():
        lbl.config(text = "Input range should be positive integers between 1 and 106")
        return

    # input range validation
    start = int(bug_start.get())
    end = int(bug_end.get())
    if start < 1 or start > 106 or end < 1 or end > 106:
        lbl.config(text = "Input range should be positive integers between 1 and 106")
        return
    elif start > end:
        lbl.config(text = "Input range invalid; start of range greater than end")
        return

    # construct bash string based on script
    global script
    script = cb.get()
    bash_str = get_bash(start, end)

    # ensure script selected
    if bash_str == "":
        lbl.config(text = "Please select a script")
        return

    # update bug count and running/progress labels
    global bug_count
    bug_count = (end - start + 1)
    lbl.config(text = f"Running \"{bash_str}\" ...")
    progress.set(0)
    ratio.config(text = f"\tComplete: 0/{bug_count} ({progress.get()}%)")

    # clear results area
    results.configure(state = "normal")
    results.delete("1.0", END)
    results.configure(state = "disabled")

    # run script in separate process
    global thread
    thread = threading.Thread(target = run, args = (start,), daemon = True)
    thread.start()

    # flag as executing and start update loop
    executing = True
    root.after(10000, update, start, end)

# init header frame
header = Frame(bg = "white")

# dropdown options
scripts = ["cloud.sh", "local.sh", "baseline.sh", "ablations.sh"]

# script combobox
cb = ttk.Combobox(header, values = scripts, state = "readonly", width = 12)
cb.set("Select")
cb.pack(side=LEFT)

# input range
from_label = Label(header, text = "From", bg = "white").pack(side = LEFT)
from_entry = Entry(header, textvariable = bug_start, width = 5).pack(side = LEFT)
to_label = Label(header, text = "To", bg = "white").pack(side = LEFT)
to_entry = Entry(header, textvariable = bug_end, width = 5).pack(side = LEFT)

# button to execute script
Button(header, text = "Execute", bg = "light blue", fg = "black", command = execute).pack(side=LEFT)

# progress bar
bar = ttk.Progressbar(header, orient = HORIZONTAL, maximum = 100.1, length = 150, variable = progress)
bar.pack(side = RIGHT)

# pad header and pack
for child in header.winfo_children():
    child.pack_configure(padx = 5, pady = 10)

bar.pack_configure(padx = 50, pady = 10) # give bar more space
header.pack()

# loader to show command and amount complete
loader = Frame(bg = "white")
lbl = Label(loader, text = " ", bg = "white")
lbl.pack(side = LEFT)
ratio = Label(loader, text = " ", bg = "white")
ratio.pack(side = LEFT)
loader.pack()

# results display area
results = st.ScrolledText(root, width = 100, height = 100)
results.configure(state = "disabled")
results.pack(side = BOTTOM)

root.mainloop()
