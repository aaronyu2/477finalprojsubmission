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

def process_row(args) :
  model, row = args
  data = prompts.to_patent_data(row, "dataset", prefer_local=False, mode=TEXT_IMAGE)
  if data is None:
    return
  messages = inference_llava.generate_prompt_llava(model, data)
  data_json = inference_llava.to_json(data, folder_path="finetuning/data/")
  return data_json
  
  
  
  

def main():
  file = 'data/hupd_sample_train_images_merged.feather'
  print("Reading file ", file)
  df = pd.read_feather(file)
  model = inference_llava.get_llava_model(keep_config_only=True)
  
  print("Beginning process")
  temp_list = []
  part_no = 0
  for i, row in tqdm(df.iterrows(), total=df.shape[0], desc="Processing rows", position=0):
    temp_list.append((model, row))
    if i % 50000 == 0:
      with multiprocessing.Pool(os.cpu_count()) as pool:
        # use tqdm
        results = list(tqdm(pool.imap(process_row, temp_list), total=len(temp_list), position=1))
        results_json = [res for res in results if res is not None]
        if len(results_json) > 0:
          with open(f"finetuning/data/part_{part_no}.json", 'w') as f:
            f.write(results_json)
          part_no += 1
      temp_list = []
  
if __name__=="__main__":
  main()  
  