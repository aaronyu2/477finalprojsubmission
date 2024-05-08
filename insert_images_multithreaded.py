import requests
from bs4 import BeautifulSoup
from typing import List
import pandas as pd
import json
import multiprocessing
from tqdm import tqdm
import os
import signal
def get_images(patent_number: str) -> str:
    """Function to fetch the image URL from Google Patents."""
    base_url = "https://patents.google.com/patent/"
    url = f"{base_url}{patent_number}/en"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            image_tags = soup.find_all('meta', {'itemprop': 'full'})
            image_urls = [tag['content'] for tag in image_tags]
            return json.dumps(image_urls)
        print(f"Failed to fetch images for {patent_number}")
    except:
        return ''
    return ''
  
if __name__ == '__main__':
    print('Reading 1Million feather file')
    df = pd.read_feather('data/hupd_all_train_merged.feather')
    print("Processing strings")

    df['publication_number_first'] = df['publication_number'].str.split('-').str[0]
    # only get the first 100 rows
    # df = df.head(100)
    publication_list = df['publication_number_first'].tolist()
    # handle this with multiprocessing, with tqdm
    with multiprocessing.Pool(processes=os.cpu_count()*2) as pool:
        image_urls = list(tqdm(pool.imap(get_images, publication_list), total=len(df)))
    print("Saving dataframe...")
    # save it back to the dataframe
    df['image_urls'] = image_urls
    df.to_feather('data/hupd_all_train_images_merged.feather')
    
    