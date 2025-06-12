# Import required libraries
import requests
import asyncio

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# API keys and endpoint
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
API_VERSION = os.getenv("API_VERSION", "2025-04-01-preview")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_ENDPOINT = os.getenv("OPENAI_API_BASE")

async def upload_file(file_name: str, file_path: str, purpose: str = "fine-tune") -> str:
    """
    Upload a file to either Azure or OpenAI based on the configuration in the .env file.

    Args:
        file_name (str): The name of the file to upload.
        file_path (str): The path to the file to upload.
        purpose (str): The purpose of the file upload (e.g., "fine-tune", "evals"). Defaults to "fine-tune".

    Returns:
        str: The file ID of the uploaded file, or an empty string if the upload fails.
    """
    use_openai = os.getenv("OAI_API_TYPE", "azure").lower() == "openai"

    if use_openai:
        # OpenAI file upload logic
        print("Using OpenAI API for file upload...")
        headers = {
            'Authorization': f'Bearer {OPENAI_API_KEY}',
        }
        with open(file_path, 'rb') as file:
            response = await asyncio.to_thread(
                requests.post,
                f'{OPENAI_API_ENDPOINT}/v1/files',
                headers=headers,
                files={"file": file},
                data={"purpose": purpose}
            )

        if response.status_code in (201, 200):
            print("File uploaded successfully to OpenAI.")
            return response.json().get('id', '')
        else:
            print(f"Failed to upload file to OpenAI. Status Code: {response.status_code}")
            print(response.text)
            return ''
    else:
        # Azure file upload logic
        with open(file_path, 'rb') as file:
            response = await asyncio.to_thread(
                requests.post,
                f'{AZURE_API_ENDPOINT}/openai/files',
                params={'api-version': f"{API_VERSION}"},
                headers={
                    'api-key': AZURE_API_KEY,
                },
                files={
                    "file": (file_name, file, 'multipart/form-data'),
                    "purpose": (None, "evals"),
                }
            )

        if response.status_code in (201, 200):
            print("File uploaded successfully to Azure.")
            return response.json().get('id', '')
        else:
            print(f"Failed to upload file to Azure. Status Code: {response.status_code}")
            print(response.text)
            return ''