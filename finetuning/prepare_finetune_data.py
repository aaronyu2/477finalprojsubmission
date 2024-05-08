# important: run this file from the root directory
import prompts
from prompts import PatentData
import pandas as pd
import multiprocessing
import finetuning.augmentations as augm
from tqdm import tqdm
from prompts import TEXT_IMAGE, TEXT_ONLY, IMAGE_ONLY
import inference_llava
import os
import json

NUM_PROCESSES = os.cpu_count()*6

def process_row(args) :
  model, row = args
  try:
    data = prompts.to_patent_data(row, "dataset", prefer_local=False, mode=TEXT_IMAGE)
    if data is None: return
    # if data.image is not None: print("HI")  
    messages = inference_llava.generate_prompt_llava(model, data)
    data_json = inference_llava.to_json(data, folder_path="finetuning/data/")
    del data
    return data_json
  except Exception as e:
    print(e)
    return None
  

def main():
  file = 'data/hupd_all_train_images_merged.feather'
  print("Reading file ", file)
  df = pd.read_feather(file)
  model = inference_llava.get_llava_model(keep_config_only=True, device="cpu")
  
  print("Beginning process")
  splits, temp_list = [], []
  for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Creating batches", position=0):
    mod = 10000
    temp_list.append((model, row))
    if i % mod == mod-1:
      splits.append(temp_list)
      temp_list = []
      
  for i, split in tqdm(enumerate(splits), total=len(splits), desc="Processing splits", position=0):
    if os.path.exists(f"finetuning/data/part_{i}.json"):
      print(f"Skipping part {i}")
      continue
    with multiprocessing.Pool(os.cpu_count()*5) as pool:
      results = list(tqdm(pool.imap(process_row, split), total=len(split), position=1))
      results_json = [res for res in results if res is not None]
      if len(results_json) > 0:
        filename = f"finetuning/data/part_{i}.json"
        print("Writing to", filename)
        with open(filename, 'w') as f:
          f.write(json.dumps(results_json))

  
if __name__=="__main__":
  main()  
  