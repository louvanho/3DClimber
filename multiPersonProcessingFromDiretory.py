import os
import sys
import subprocess

from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--videodirectory", type=str, default=None, help="Directory with the videos to process")
    parser.add_argument("--directory", type=str, default=None, help="Directory to store the processed files")
    
    args = parser.parse_args()

    if args.videodirectory is None:
        print("Please provide a directory with the videos to process")
        sys.exit(1)
    if args.directory is None:
        print("Please provide a directory to store the processed files")
        sys.exit(1)
    
    if not os.path.exists(args.directory):
        os.makedirs(args.directory)
        print(f"Created directory: {args.directory}")

    for file in os.listdir(args.videodirectory):
        if file.endswith(".mp4") or file.endswith(".MP4") or file.endswith(".mov") or file.endswith(".MOV"):
            print(f"Processing file: {file}")
            command = f"python multiPersonProcessing.py --video {os.path.join(args.videodirectory, file)} --directory {os.path.join(args.directory, file)} --copyvideo"
            print(command)
            result = subprocess.run(command, shell=True)
            if result.returncode != 0:
                print(f"Error processing file: {file}. Return code: {result.returncode}")
                sys.exit(1)
            print(f"Processed file: {file}")

    print("All files processed")
    sys.exit(0)