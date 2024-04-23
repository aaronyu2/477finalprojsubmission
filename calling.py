# Pretty print
from pprint import pprint
# Datasets load_dataset function
from datasets import load_dataset
# Transformers Autokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# Standard PyTorch DataLoader
from torch.utils.data import DataLoader

from PIL import Image as PILImage

from prompts import PatentData

from openai import AzureOpenAI
api_base = 'https://gerstein-westus.openai.azure.com/'
api_key="b4521b5444d74d0284e9aa797245b25d"
deployment_name = 'gpt4-vision-preview'
api_version = '2023-12-01-preview'

client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    base_url=f"{api_base}openai/deployments/{deployment_name}/extensions",
)

def call_openai(data: PatentData, model=deployment_name):
    """
    Calls OpenAI with the given data and model.
    Returns:
        response: the response from the OpenAI API
    """
    messages = generate_prompt_openai(data)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return response


from vertexai.generative_models import Image as VertexImage
from vertexai.generative_models import GenerativeModel, Part
import urllib.request
import http.client
import typing
vertexai.init(project="cs477-final-project", location="us-central1")


def call_gemini(data: PatentData):
    vertex_model = GenerativeModel("gemini-1.0-pro-vision")
    response = vertex_model.generate_content(messages)
    return response



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