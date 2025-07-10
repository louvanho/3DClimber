import subprocess
import sys

def download_urls(file_path, download_directory):
    with open(file_path, 'r') as file:
        urls = file.readlines()
    
    for url in urls:
        url = url.strip()
        print(f'Downloading {url} into {download_directory}')
        if 'c01' in url:
            subprocess.run(['wget', '-P', download_directory, url])

if __name__ == "__main__":
    download_directory = sys.argv[1]
    file_path = 'aist_group.txt'  # Replace with the path to your txt file
    download_urls(file_path, download_directory)