import os
import pandas as pd
import json

BASE_DIRECTORY = "data/"

def save_file_folder(contents,
                     basepath: str,
                     filename: str,
                     depth:int =4) -> str:
    # split the first depth*3 characters of the filename into a folder structure, 3 characters per folder
    # e.g. filename = '238297d28e8e
    # basepath = 'data'
    # depth = 4
    # folder structure = 'data/238/297/d28/e8e/'
    # filename = '238297d28e8e'
    
    temp_folder = ''
    for i in range(depth):
        temp_folder += filename[i*3:(i+1)*3] + '/'
    folder = os.path.join(basepath, temp_folder)
    os.makedirs(folder, exist_ok=True)
    if type(contents) == str:
        with open(os.path.join(folder, filename), 'w') as f:
            f.write(contents)
    else:
        with open(os.path.join(folder, filename), 'wb') as f:
            f.write(contents)
    return str(os.path.join(folder, filename))

def get_server_subdirectory(url: str) -> str:
    return url.replace('https://patentimages.storage.googleapis.com/', '')
    

if __name__=="__main__":
    if not os.path.exists('data'):
        os.mkdir('data')
    
    df = pd.read_feather('/home/ubuntu/cs477-final-project/data/hupd_sample_train_images_merged.feather')
    image_urls = []
    df_image_urls = df['image_urls']
    for image_url in df_image_urls:
        image_urls.extend(json.loads(image_url))
    image_urls = list(set(image_urls))
    print("Total images to download: ", len(image_urls))
    
    # download this across multiple threads and multiple processes
    # url looks like https://patentimages.storage.googleapis.com/23/82/97/d28e8e17f1597a/US20140002137A1-20140102-D00002.png
    from concurrent.futures import ThreadPoolExecutor
    import requests
    from tqdm import tqdm
    import time
    import random
    import threading
    import signal
    import sys
    
    def signal_handler(sig, frame):
        print("Exiting...")
        sys.exit(0)
    signal.signal(signal.SIGINT, signal_handler)
    
    def download_image(url: str) -> str:
        response = requests.get(url)
        if response.status_code == 200:
            server_subdir = save_file_folder(response.content, BASE_DIRECTORY, url.split('/')[-1])
            return True
        print("Warning: image does not exist: ", url)
        return False
    
    def download_images(image_urls: list):
        with ThreadPoolExecutor(max_workers=16) as executor:
            # use tqdm and threadpoolexecutor to download images
            futures = []
            for url in tqdm(image_urls, desc="Queueing tasks"):
                futures.append(executor.submit(download_image, url))
            for future in tqdm(futures):
                future.result()
    download_images(image_urls)
    print("Downloaded images to data folder")
    
    
    