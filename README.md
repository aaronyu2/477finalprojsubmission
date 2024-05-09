# CPSC 477 Final Project - Multimodal NLP for Patent Documents

## Team Members
- Bill Qian
- Katherine He
- Aaron 

## Running the project
### Prerequesites
To run this project, we recommend using Python 3.10
1. Clone the repository
2. Run `pip install -r requirements.txt`
3. Clone and install the LLaVa project
```python
git clone https://github.com/haotian-liu/LLaVA.git
cd LLaVA
# create a new conda environment and install LLaVA
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
# Install things needed for training
pip install -e ".[train]"
pip install flash-attn --no-build-isolation
```

### Download the datasets
You may run through the 
Our [data]() is also available to download. This does not, however, include the images, which can be downloaded by running `PYTHON

### Consolidating/generating images


## Finetuning

In order to fine-tune the model, we will first need to precompute collages and along with the prompts, save them to disk. We follow the LLaVa prompt format, which is as follows:

```json
[
  {
    "id": "e8bac9e9-cc56-4a31-8e6c-5e71308a7c56",
    "image": "finetuning/data/e8bac9e9-cc56-4a31-8e6c-5e71308a7c56.jpg",
    "conversations": [
      {
        "from": "human",
        "value": "<image>\nClassify the following patent into a class in the International Patent Classification system. Answer with the a 4-character IPC class, do not answer with anything else"
      },
      {
        "from": "gpt",
        "value": "F08H"
      },
    ]
  },
]
```
Start the finetuning process by running the `finetune_lora.sh` script. We trained on 8xA10G (192GB VRAM total) on an AWS g5.48xlarge instance. You may need to adjust batch size and gradient accumulation steps (in inverse proportions) in case this does not fit into VRAM.