# Pretty print
from pprint import pprint
# Datasets load_dataset function
from datasets import load_dataset
# Transformers Autokenizer
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
# Standard PyTorch DataLoader
from torch.utils.data import DataLoader
from botocore.exceptions import ClientError

from PIL import Image as PILImage

from prompts import PatentData
import json
from openai import AzureOpenAI
api_base = 'https://DEPLOYNAME.openai.azure.com/'
api_key="YOUR_KEY_HERE"
deployment_name = 'gpt4-vision-preview'
api_version = '2023-12-01-preview'

azure_client = AzureOpenAI(
    api_key=api_key,  
    api_version=api_version,
    base_url=f"{api_base}openai/deployments/{deployment_name}/extensions",
)

def call_openai(data: PatentData, model=deployment_name, return_token_usage=False):
    """
    Calls OpenAI with the given data and model.
    Returns:
        response: the response from the OpenAI API
    """
    if data.messages is None:
        raise ValueError("PromptData object must have messages attribute set.")
    response = azure_client.chat.completions.create(
        model=model,
        messages=data.messages,
        max_tokens=10
    )
    if return_token_usage:
        return response.choices[0].message.content, response.usage.total_tokens
    return response.choices[0].message.content


from vertexai.generative_models import Image as VertexImage
from vertexai.generative_models import GenerativeModel, Part
import urllib.request
import http.client
import vertexai
import typing
vertexai.init(project="cs477-final-project", location="us-central1")


def call_gemini(data: PatentData):
    vertex_model = GenerativeModel("gemini-1.0-pro-vision")
    response = vertex_model.generate_content(messages)
    return response.text



def concat_images(images):
    """ Concatenates a list of PIL images horizontally. """
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


import boto3
boto3_client = boto3.client(
    service_name="bedrock-runtime", region_name="us-east-1"
)

def call_anthropic(data: PatentData):
    """ Calls the Anthropic Claude 3 Haiku model with the given data. """
    if data.messages is None:
        raise ValueError("PromptData object must have messages attribute set.")

    model_id = "anthropic.claude-3-haiku-20240307-v1:0"
    request_body = {
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 10,
        "messages": data.messages
    }

    try:
        response = boto3_client.invoke_model(
            modelId=model_id,
            body=json.dumps(request_body),
        )

        # Process and print the response
        result = json.loads(response.get("body").read())
        input_tokens = result["usage"]["input_tokens"]
        output_tokens = result["usage"]["output_tokens"]
        output_list = result.get("content", [])
        res1 = ""
        for output in output_list:
            res1 += output["text"]

        return res1
    except ClientError as err:
        print(err)
        raise


