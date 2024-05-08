# Pretty print
from pprint import pprint
from PIL import Image as PILImage
from dataclasses import dataclass
from typing import List, Dict
import base64
from mimetypes import guess_type
import json
import math
from PIL import Image
import urllib
import os
import io
from typing import Optional
import uuid


TEXT_IMAGE = 0
TEXT_ONLY = 1
IMAGE_ONLY = 2
@dataclass
class PatentData:
    text: str
    image: PILImage
    label: str
    model_output: str
    messages: any
    model_type: str
    mode: int    
    original_images: Optional[List[str]] = None
    original_data_obj: Optional[Dict] = None

    
def load_image_from_url(url: str) -> PILImage:
    try:
        response = urllib.request.urlopen(url)
        image = PILImage.open(response)
        return image
    except Exception as e:
        return None

def image_collage(images: List[str]) -> PILImage:
    """
    Collage multiple images into a single image.
    """
    # Load the images
    def load_image(image):
        if "http" in image:
            return load_image_from_url(image)
        try:
            return PILImage.open(image)
        except:
            return None
    images = [load_image(image) for image in images]
    images = [image for image in images if image is not None]
    collage = create_collage(images)
    
    return collage



def to_patent_data(data,
                   model_type: str,
                   prefer_local=False,
                   mode = TEXT_IMAGE,
                   include_original_data=False
    ):
    # print(data["cpc_labels"])
    target = data["ipc_label"][:4]
    
    def get_folder_path_from_file(file: str) -> str:
        temp_folder = ''
        for i in range(4):
            temp_folder += file[i*3:(i+1)*3] + '/'
        folder = os.path.join(temp_folder, file)
        return folder
    image_urls = json.loads(data["image_urls"])
    if prefer_local:
        image_url_names = [get_folder_path_from_file(url.split('/')[-1]) for url in image_urls]
        image_urls = [f"images/data/{name}" for name in image_url_names]
    # print(image_urls)
    if mode == IMAGE_ONLY and len(image_urls) == 0:
        return None
    # print(mode, (mode == TEXT_IMAGE or mode == IMAGE_ONLY), image_urls)
    ret =  PatentData(
        text= data["abstract"],
        image = image_collage(image_urls) if (mode == TEXT_IMAGE or mode == IMAGE_ONLY) else None,
        label = target,
        model_type=model_type,
        model_output=None,
        messages=None,
        mode=mode
    )
    if include_original_data:
        ret.original_data_obj = data
        ret.original_images = image_urls
    return ret

def loss_function():
    pass

def concat_images_dp(images):
    # TODO: finish this algorithm later
    dp = [[0] * len(images) for _ in range(len(images))]
    
def create_collage(image_list):
    # Find the number of images
    if len(image_list) == 0:
        # return empty 480 x 480 image
        return Image.new('RGB', (480, 480))
    
    num_images = len(image_list)

    # Calculate the number of rows and columns for the collage
    rows = math.ceil(math.sqrt(num_images))
    cols = math.ceil(num_images / rows)

    # Find the smallest image, and save its size
    min_img_width = min_img_height = float('inf')
    for image in image_list:
        if image.size[0] < min_img_width:
            min_img_width = image.size[0]
        if image.size[1] < min_img_height:
            min_img_height = image.size[1]

    # Create a new image that will be the collage
    collage_width = cols * min_img_width
    collage_height = rows * min_img_height
    collage = Image.new('RGB', (collage_width, collage_height))

    # Iterate over the images and paste them into the collage
    for index, image in enumerate(image_list):
        # Calculate the position in the collage where the image will be pasted
        pos_x = (index % cols) * min_img_width
        pos_y = (index // cols) * min_img_height

        # Resize the image and paste it into the collage
        image = image.resize((min_img_width, min_img_height))
        collage.paste(image, (pos_x, pos_y))
    
    # resize to at most 3000x3000, KEEP ASPECT RATIO
    max_size = 3000
    if collage.size[0] > max_size or collage.size[1] > max_size:
        collage.thumbnail((max_size, max_size))
    

    return collage


def concat_images(images):
    if len(images) == 1:
        return images[0]
    if len(images) < 1:
        return None
    max_height = 0
    total_width = 0
    for image in images:
        width, height = image.size
        max_height = max(height, max_height)
        max_width += width
    to_return = PILImage.new("RGB", (total_width, max_height))
    curr_x = 0
    for image in images:
        to_return.paste(image, (curr_x, 0))
        curr_x += image.size[0]
    # TODO maybe save as image file
    return to_return
  

def PIL_image_to_data_url(image, return_formatted=True, image_format="png"):
    # Save the image to a temporary file io.BytesIO object
    temp_file = io.BytesIO()
    image.save(temp_file, format=image_format.upper())
    # Encode the image file into a data URL
    mime_type = "image/"+image_format
    base64_encoded_data = base64.b64encode(temp_file.getvalue()).decode('utf-8')
    if return_formatted: return f"data:{mime_type};base64,{base64_encoded_data}"
    else: return base64_encoded_data

def generate_prompt_openai(data: PatentData):
    """
    Returns:
        messages: list to use as input to client.chat.completions.create()
    Note that the data object is modified in place.
    """
    
    messages = [
        { "role": "user", "content": "Classify the following patent into a class in the International Patent Classification system. Answer with the a 4-character IPC class, do not answer with anything else"},
    ]
    mode = data.mode
    prompt_block = []
    if mode == TEXT_IMAGE or mode == TEXT_ONLY:
        prompt_block.append({ 
            "type": "text", 
            "text": data.text
        })
    if mode == TEXT_IMAGE or mode == IMAGE_ONLY:
        prompt_block.append({ 
            "type": "image_url",
            "image_url": {
                "url": PIL_image_to_data_url(data.image)
            }
        })
    messages.append({ "role": "user", "content": prompt_block })
    data.messages = messages
    return messages

    

def generate_prompt_gemini(data: PatentData):
    # warn if data.image is not a PIL image
    if not isinstance(data.image, PILImage):
        raise ValueError("data.image must be a PIL image")
    messages = [
        data.image, # note that PIL images are supported here somehow
        "Classify the following patent into a class in the International Patent Classification system. Answer with the a 4-character IPC class.\n" + data.text
    ]
    data.messages = messages
    return messages

def generate_prompt_claude(data: PatentData):
    
    mode = data.mode
    prompt_block = [{ 
        "type": "text", 
        "text": "Classify the following patent into a class in the International Patent Classification system. Answer with the a 4-character IPC class. Do not answer with anything else, just the 4-character class.\n"
    }]
    if mode == TEXT_IMAGE or mode == TEXT_ONLY:
        prompt_block.append({ 
            "type": "text", 
            "text": data.text
        })
    if mode == TEXT_IMAGE or mode == IMAGE_ONLY:
        prompt_block.append({ 
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": PIL_image_to_data_url(data.image, return_formatted=False, image_format="jpeg"),
            },
        })
    messages = [
        { "role": "user", "content": prompt_block} 
    ]
    data.messages = messages
    return messages