import os
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from dataclasses import dataclass
from prompts import PatentData

os.environ["HF_DATASETS_CACHE"] = "./hf_cache"
os.environ["HF_HOME"] = "./hf_cache"

MODEL_PATH = "liuhaotian/llava-v1.5-7b"

@dataclass
class LocalLLaVaModel:
    tokenizer: any
    model: any
    image_processor: any
    context_len: int
  

def get_llava_model(model_path=MODEL_PATH) -> LocalLLaVaModel:
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    return LocalLLaVaModel(tokenizer, model, image_processor, context_len)
  
def infer_conv_mode(model_name):
    if "llama-2" in model_name.lower():
        conv_mode = "llava_llama_2"
    elif "mistral" in model_name.lower():
        conv_mode = "mistral_instruct"
    elif "v1.6-34b" in model_name.lower():
        conv_mode = "chatml_direct"
    elif "v1" in model_name.lower():
        conv_mode = "llava_v1"
    elif "mpt" in model_name.lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    return conv_mode

from llava.conversation import conv_templates, SeparatorStyle

def generate_prompt_llava(model_obj: LocalLLaVAModel, data: PatentData):
    conv = conv_templates[infer_conv_mode(model_path)].copy()
    if model_obj.image is not None:
        if model_obj.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        model_obj.image = None
    else:
        conv.append_message(conv.roles[0], inp)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    data.messages = prompt
    return prompt

from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def inference_llava(model_obj:LocalLLaVaModel, data: PatentData):
    if data.messages is None:
        raise ValueError("No messages found in the data.")
      
    image = data.image
    image_tensor = process_images([image], image_processor, model.config) # 
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(data.messages, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    
    with torch.inference_mode():
        output_ids = model.generate(
          input_ids, 
          images = image_tensor,
          image_sizes = [image_size],
          temperature = 0.7,
          max_new_tokens = 2048,
          use_cache = True
        )
    outputs = tokenizer.decode(output_ids[0]).strip()
    return outputs    