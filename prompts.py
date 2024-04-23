# Pretty print
from pprint import pprint
from datasets import load_dataset
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
from torch.utils.data import DataLoader
from PIL import Image as PILImage
from dataclasses import dataclass
from typing import List, Dict
import base64
from mimetypes import guess_type

@dataclass
class PatentData:
    text: str
    image: PILImage
    label: int
    messages: any

def to_patent_data(data):
    return PatentData(data["text"], data["image"], data["label"], data["messages"])


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
  

def PIL_image_to_data_url(image):
    # Save the image to a temporary file io.BytesIO object
    temp_file = io.BytesIO()
    image.save(temp_file, format="PNG")
    # Encode the image file into a data URL
    mime_type = "image/png"
    base64_encoded_data = base64.b64encode(temp_file.getvalue()).decode('utf-8')
    return f"data:{mime_type};base64,{base64_encoded_data}"

def generate_prompt_openai(data: PatentData):
    """
    Returns:
        messages: list to use as input to client.chat.completions.create()
    Note that the data object is modified in place.
    """
    messages = [
        { "role": "user", "content": "Classify the following patent into a class in the International Patent Classification system. Answer with the a 4-character IPC class."},
        { "role": "user", "content": [  
            { 
                "type": "text", 
                "text": data.text
            },
            { 
                "type": "image_url",
                "image_url": {
                    "url": PIL_image_to_data_url(data.image)
                }
            }
        ] } 
    ]
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
