import sys
import subprocess

# Prepare multiple training experiments in batch file
p = subprocess.Popen('run_training.bat', shell=True, stdout = subprocess.PIPE)

stdout, stderr = p.communicate()

print(p.returncode)
