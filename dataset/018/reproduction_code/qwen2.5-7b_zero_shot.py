import subprocess

package_name = "."
subprocess.run(["python", "-m", "pip", "install", "--use-feature=2020-resolver", package_name])