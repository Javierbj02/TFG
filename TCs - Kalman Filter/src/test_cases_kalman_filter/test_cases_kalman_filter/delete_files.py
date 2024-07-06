# I want a python script that deletes all files that are inside a directory.  It must delete only files, not folders.  That include files inside folders that are inside the main folder.
# The main folder is the one that the user will provide as input.

import os
import shutil
import sys




def delete_files(directory):
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)


            if os.path.isfile(file_path):
                os.remove(file_path)

                
if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python delete_files.py <directory>")
        sys.exit(1)

    directory = sys.argv[1]
    delete_files(directory)