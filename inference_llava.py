import os
import torch
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model
from dataclasses import dataclass
from prompts import PatentData
from PIL import Image as PILImage
from prompts import TEXT_IMAGE, TEXT_ONLY, IMAGE_ONLY


os.environ["HF_DATASETS_CACHE"] = "./hf_cache"
os.environ["HF_HOME"] = "./hf_cache"

MODEL_PATH = "liuhaotian/llava-v1.5-7b"

@dataclass
class LocalLLaVaModel:
    tokenizer: any
    model_path: any
    model: any
    image_processor: any
    context_len: int

@dataclass
class DummyModel:
    config: any
  

def get_llava_model(model_path=MODEL_PATH, keep_config_only=False) -> LocalLLaVaModel:
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=get_model_name_from_path(model_path)
    )
    if keep_config_only:
        import copy
        model = DummyModel(config=copy.deepcopy(model.config))
    return LocalLLaVaModel(tokenizer, model_path, model, image_processor, context_len)
  
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

def generate_prompt_llava(model_obj: LocalLLaVaModel, data: PatentData):
    conv = conv_templates[infer_conv_mode(model_obj.model_path)].copy()
    inp = "Classify the following patent into a class in the International Patent Classification system. Answer with the a 4-character IPC class. Do not answer with anything else, just the 4-character IPC class. \n"
    mode = data.mode
    if mode == TEXT_IMAGE:
        if data.image is None:
            raise ValueError("Image must be provided for TEXT_IMAGE mode.")
        if model_obj.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp + data.text
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp + data.text
        conv.append_message(conv.roles[0], inp)
        data.image = None
    elif mode == IMAGE_ONLY:
        if data.image is None:
            raise ValueError("Image must be provided for IMAGE_ONLY mode.")
        if model_obj.model.config.mm_use_im_start_end:
            inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
        else:
            inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
        conv.append_message(conv.roles[0], inp)
        data.image = None
    elif mode == TEXT_ONLY:
        conv.append_message(conv.roles[0], inp + data.text)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    data.messages = prompt
    return prompt

from llava.mm_utils import process_images, tokenizer_image_token, get_model_name_from_path

def make_blank_PIL_image():
    return PILImage.new('RGB', (128, 128), (255, 255, 255))

def inference_llava(model_obj:LocalLLaVaModel, data: PatentData):
    if data.messages is None:
        raise ValueError("No messages found in the data.")
    model = model_obj.model
    image = data.image
    tokenizer = model_obj.tokenizer
    conv = data.messages
    if image is None:
        image = make_blank_PIL_image()
    image_tensor = process_images([image], model_obj.image_processor, model.config) # 
    if type(image_tensor) is list:
        image_tensor = [image.to(model.device, dtype=torch.float16) for image in image_tensor]
    else:
        image_tensor = image_tensor.to(model.device, dtype=torch.float16)
    input_ids = tokenizer_image_token(data.messages, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(model.device)
    # stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    # keywords = [stop_str]
    image_size = image.size
    with torch.inference_mode():
        output_ids = model.generate(
          input_ids, 
          images = image_tensor,
          image_sizes = [image_size],
          temperature = 0.7,
          max_new_tokens = 20,
          use_cache = True
        )
    outputs = tokenizer.decode(output_ids[0]).strip()
    return outputs    

def uuid_to_folder_path(uuid:str):
    # if uuid was abcdefg, then the folder path should bve
    # a/b/c/defg...
    temp_folder = ''
    for i in range(4):
        temp_folder += uuid[i] + '/'
    folder = os.path.join(temp_folder, uuid)
    return folder


def to_json(data:PatentData, folder_path:str):
        # json should look like this:
        ## [
#   {
#     "id": "997bb945-628d-4724-b370-b84de974a19f",
#     "image": "part-000001/997bb945-628d-4724-b370-b84de974a19f.jpg",
#     "conversations": [
#       {
#         "from": "human",
#         "value": "<image>\nWrite a prompt for Stable Diffusion to generate this image."
#       },
#       {
#         "from": "gpt",
#         "value": "a beautiful painting of chernobyl by nekro, pascal blanche, john harris, greg rutkowski, sin jong hun, moebius, simon stalenhag. in style of cg art. ray tracing. cel shading. hyper detailed. realistic. ue 5. maya. octane render. "
#       },
#     ]
#   }
    obj = {}
    # generate a uuid
    uuid = str(uuid.uuid4())
    obj["id"] = uuid
    save_path = os.path.join(folder_path, uuid_to_folder_path(uuid) + ".jpg")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    data.image.save(save_path)
    
    obj["image"] = save_path
    
    conversation = [
        {
            "from": "human",
            "value": data.messages
        },
        {
            "from": "gpt",
            "value": data.label
        }
    ]
    obj["conversations"] = conversation
    
    