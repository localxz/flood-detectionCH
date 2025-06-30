"""
Grabs file names from a folder and then saves them to a text file.
"""

import os
import sys


def get_file_names(folder_path, output_file="file_list.txt"):

    try:
        if not os.path.exists(folder_path):
            print(f"Error: Folder '{folder_path}' does not exist.")
            return

        if not os.path.isdir(folder_path):
            print(f"Error: '{folder_path}' is not a directory.")
            return

        files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

        files.sort()

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"Files in folder: {folder_path}\n")
            f.write(f"Total files: {len(files)}\n")
            f.write("-" * 50 + "\n\n")

            for file in files:
                f.write(file + "\n")

        print(f"Successfully saved {len(files)} file names to '{output_file}'")

    except PermissionError:
        print(f"Error: Permission denied accessing '{folder_path}'")
    except Exception as e:
        print(f"An error occurred: {e}")


def main():
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
    else:
        folder_path = input("Enter folder path (or press Enter for current directory): ").strip()
        if not folder_path:
            folder_path = "."

    output_file = "file_list.txt"

    get_file_names(folder_path, output_file)


if __name__ == "__main__":
    main()