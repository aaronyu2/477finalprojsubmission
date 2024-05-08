import json
import os

all_objs = []

def image_exists(datapoint):
  filename = datapoint['image']
  return os.path.exists(filename)
    

for i in range(0, 100000000):
  if not os.path.exists(f"finetuning/data/part_{i}.json"):
    break
  
  with open(f"finetuning/data/part_{i}.json", 'r') as f:
    data = json.load(f)
    
    data = [obj for obj in data if image_exists(obj)]
    all_objs.extend(data)
    

with open("finetuning/data/all_data.json", "w") as f:
  f.write(json.dumps(all_objs[:100000]))
