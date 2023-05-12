import requests
import shutil
from tqdm.auto import tqdm
import os
#we store the files on Dropbox. If the following code does not
#work for some reason, you can directly download them
dataset_url = "https://www.dropbox.com/s/zwz6v7ysglenmg9/dataset.zip?dl=1"
tracks_url = "https://www.dropbox.com/s/memj8ws34i87rhs/tracking_results.zip?dl=1"
sota_tracks = "https://www.dropbox.com/s/yogbm7rwfrhl3wt/sota_tracks_multiple_droplets.zip?dl=1"

def download_file(url,dest_dir="./"):
    file_path = f"{dest_dir}/temp.zip"
    with requests.get(url, stream=True) as r:
        #get length
        total_length = int(r.headers.get("Content-Length"))

        #progress bar via tqdm
        with tqdm.wrapattr(r.raw, "read", total=total_length, desc="") as file:
            #save the file
            with open(file_path, 'wb') as output:
                shutil.copyfileobj(file, output)

    #unzip the file
    shutil.unpack_archive(file_path, dest_dir)
    #remove the zip file since not needed
    if os.path.exists(file_path):
        os.remove(file_path)

    print(f"If download not successful, use {url} to download directly from dropbox...")

def run():
    download_file(dataset_url)
    download_file(tracks_url)
    download_file(sota_tracks)

if __name__ == "__main__":
    run()
