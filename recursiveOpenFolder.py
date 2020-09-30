# https://stackoverflow.com/questions/2578022/python-2-5-2-trying-to-open-files-recursively
# https://stackoverflow.com/questions/15450733/open-a-python-file-in-notepad-from-a-program
# https://stackoverflow.com/questions/1176441/how-to-filter-files-with-known-type-from-os-walk
import os
import subprocess

PATH = "C:/Users/X/Dropbox/GitHub/ByzShield"

for path, dirs, files in os.walk(PATH):
    # print(path)
    
    # files = [ file for file in files if file.endswith(('.py','.sh','.txt')) or file == "hosts_address_local"]
    files = [ file for file in files if file.endswith(('.py','.sh','.txt')) or file == "hosts_address_local"]
    
    for filename in files:
        fullpath = os.path.join(path, filename)
        
        # option 1: explicit
        subprocess.call(["C:/Program Files (x86)/Notepad++/notepad++.exe", fullpath])
        
        # option 2: use this if all files are already associated with notepad++
        # os.startfile(fullpath, 'notepad++.exe')
