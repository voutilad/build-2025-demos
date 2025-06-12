import time
import pandas as pd
import json
import time
from pathlib import Path
from tqdm import tqdm
from openai import AzureOpenAI

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# API keys and endpoint
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT")
API_VERSION = os.getenv("API_VERSION", "2025-04-01-preview")

def distill_from_teacher_model(teacher_model: str, input_path: str, output_path: str, max_records: int = 1000, retries: int = 3, retry_delay: int = 10):
    """
    Distill data from a teacher model and save the completions to an output file.

    Args:
        teacher_model (str): The teacher model to use for distillation.
        input_path (str): Path to the input JSONL file containing prompts.
        output_path (str): Path to the output JSONL file to save distilled data.
        max_records (int): Maximum number of records to process. Default is 1000.
        retries (int): Number of retries for failed requests. Default is 3.
        retry_delay (int): Delay (in seconds) between retries. Default is 10.
    """
    client = AzureOpenAI(azure_endpoint=AZURE_API_ENDPOINT, api_key=AZURE_API_KEY, api_version=API_VERSION)

    # Load prompts
    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f][:max_records]

    print(f"🔹 Loaded {len(data)} records for distillation")

    # Ensure the output file exists
    if not output_path.exists():
        output_path.touch()

    # Process each record
    for i, row in enumerate(tqdm(data, desc="Distilling from teacher model")):
        prompt = row["prompt"]

        for attempt in range(retries):
            try:
                print(f"\n▶️ Record {i+1}/{max_records} — Attempt {attempt+1}")

                response = client.chat.completions.create(
                    model=teacher_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    max_completion_tokens=10240
                )

                stored_id = response.id
                completion_text = response.choices[0].message.content.strip()

                # Check if completion text is empty
                if not completion_text:
                    print("⚠️ Empty completion text")
                    continue

                record = {
                    "prompt": prompt,
                    "stored_completion_id": stored_id,
                    "preferred_output": completion_text
                }

                # Append the record to the output file
                with open(output_path, "a", encoding="utf-8") as f:
                    json.dump(record, f)
                    f.write("\n")

                print(f"✅ Stored ID: {stored_id}")
                break

            except Exception as e:
                print(f"⚠️ Error on record {i+1}, attempt {attempt+1}: {e}")
                time.sleep(retry_delay)

        else:
            print(f"❌ Skipping record {i+1} after {retries} failed attempts")

    print(f"\n✅ Done. Completions saved to: {output_path}")


# function to convert the output file to SFT format
def convert_to_sft_format(input_path: str, output_path: str, system_prompt: str):
    """
    Convert the distilled data to SFT format with an optional system prompt.

    Args:
        input_path (str): Path to the input JSONL file containing distilled data.
        output_path (str): Path to the output JSONL file to save SFT-ready data.
        system_prompt (str): The system prompt to include in the SFT format.
    """
    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load distilled completions
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    print(f"🔹 Loaded {len(data)} records for SFT conversion")

    # Convert to SFT format
    sft_data = []
    for entry in data:
        prompt = entry["prompt"].strip()
        response = entry["preferred_output"].strip()

        sft_data.append({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
        })

    # Save output
    with open(output_path, "w", encoding="utf-8") as f:
        for item in sft_data:
            json.dump(item, f)
            f.write("\n")

    print(f"✅ SFT-ready file saved to: {output_path} — {len(sft_data)} records")


from pathlib import Path
import json
import time
from tqdm import tqdm
from openai import AzureOpenAI  # Make sure you're using the latest OpenAI SDK

def distill_multiple_completions_with_logprob_summary(
    teacher_model: str,
    input_path: str,
    output_path: str,
    max_records: int = 1000,
    retries: int = 3,
    retry_delay: int = 10,
    num_completions: int = 3,
    temperature: float = 0.7
):
    """
    Distill multiple completions from a teacher model with total and average logprob summaries.

    Args:
        teacher_model (str): Model name (e.g., gpt-4-turbo-2024-04-09).
        input_path (str): Path to input JSONL file with prompts.
        output_path (str): Path to output JSONL file to save completions.
        max_records (int): Maximum records to process.
        retries (int): Number of retries for failed requests.
        retry_delay (int): Delay between retries (in seconds).
        num_completions (int): Number of completions per prompt.
        temperature (float): Sampling temperature for generation.
    """

    client = AzureOpenAI(azure_endpoint=AZURE_API_ENDPOINT, api_key=AZURE_API_KEY, api_version="2024-06-01")

    input_path = Path(input_path)
    output_path = Path(output_path)

    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f][:max_records]

    print(f"🔹 Loaded {len(data)} records for distillation")

    if not output_path.exists():
        output_path.touch()

    for i, row in enumerate(tqdm(data, desc="Distilling from teacher model (summary only)")):
        prompt = row["prompt"]

        for attempt in range(retries):
            try:
                print(f"\n▶️ Record {i+1}/{max_records} — Attempt {attempt+1}")

                response = client.chat.completions.create(
                    model=teacher_model,
                    messages=[{"role": "user", "content": prompt}],
                    stream=False,
                    temperature=temperature,
                    max_completion_tokens=10240,
                    n=num_completions,
                    logprobs=True
                )

                completions = []
                stored_ids = []

                for choice in response.choices:
                    completion_text = choice.message.content.strip()

                    if not completion_text or not choice.logprobs or not choice.logprobs.content:
                        continue

                    # Collect all token logprobs
                    token_logprobs = [
                        token_info.logprob for token_info in choice.logprobs.content
                        if token_info.logprob is not None
                    ]

                    if not token_logprobs:
                        continue

                    total_logprob = sum(token_logprobs)
                    average_logprob = total_logprob / len(token_logprobs)

                    completions.append({
                        "text": completion_text,
                        "total_logprob": total_logprob,
                        "average_logprob": average_logprob
                    })

                    stored_ids.append(response.id)

                if not completions:
                    print("⚠️ No valid completions found.")
                    continue

                record = {
                    "prompt": prompt,
                    "stored_completion_ids": stored_ids,
                    "outputs": completions
                }

                with open(output_path, "a", encoding="utf-8") as f:
                    json.dump(record, f)
                    f.write("\n")

                print(f"✅ Stored {len(completions)} completions with summary logprobs.")
                break

            except Exception as e:
                print(f"⚠️ Error on record {i+1}, attempt {attempt+1}: {e}")
                time.sleep(retry_delay)

        else:
            print(f"❌ Skipping record {i+1} after {retries} failed attempts")

    print(f"\n✅ Done. Distilled completions (logprob summary) saved to: {output_path}")



def prepare_autograder_input_from_distilled_file(
    distilled_input_path: str,
    autograder_output_path: str
):
    """
    Prepare an autograder input file from distilled completions.

    Args:
        distilled_input_path (str): Path to the distilled JSONL file with prompt and outputs (with avg logprobs).
        autograder_output_path (str): Path to save the autograder input JSONL file.
    """

    distilled_input_path = Path(distilled_input_path)
    autograder_output_path = Path(autograder_output_path)

    if not distilled_input_path.exists():
        raise FileNotFoundError(f"Distilled input file not found: {distilled_input_path}")

    with open(distilled_input_path, "r", encoding="utf-8") as infile, \
         open(autograder_output_path, "w", encoding="utf-8") as outfile:

        for line in tqdm(infile, desc="Preparing autograder input"):
            record = json.loads(line)

            prompt = record.get("prompt")
            outputs = record.get("outputs", [])

            if not prompt or len(outputs) < 3:
                print(f"⚠️ Skipping record due to missing prompt or insufficient completions.")
                continue

            # Sort outputs by descending average_logprob (highest confidence first)
            sorted_outputs = sorted(outputs, key=lambda x: x.get("average_logprob", float('-inf')), reverse=True)

            autograder_record = {
                "prompt": prompt,
                "choice1": sorted_outputs[0]["text"],
                "choice2": sorted_outputs[1]["text"],
                "choice3": sorted_outputs[2]["text"]
            }

            json.dump(autograder_record, outfile)
            outfile.write("\n")

    print(f"\n✅ Autograder input saved to: {autograder_output_path}")

import json
import pandas as pd
from IPython.display import display

def preview_fine_tuning_dataset(file_path, num_samples=3, max_len=500):
    """
    Reads and displays a preview of a fine-tuning dataset in JSONL format.

    Args:
        file_path (str): Path to the fine-tuning dataset file (JSONL format).
        num_samples (int): Number of samples to display. Defaults to 3.
        max_len (int): Maximum length for truncating text fields. Defaults to 500.

    Returns:
        None
    """
    records = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for _ in range(num_samples):
            line = f.readline()
            if not line:
                break
            data = json.loads(line)
            messages = data.get("messages", [])
            
            user_message = next((m["content"] for m in messages if m["role"] == "user"), "")
            assistant_message = next((m["content"] for m in messages if m["role"] == "assistant"), "")
            
            records.append({
                "User Prompt": user_message.strip(),
                "Assistant Response": assistant_message.strip()
            })

    # Create a DataFrame
    df = pd.DataFrame(records)

    # Truncate long text for better display
    def truncate(text):
        return text if len(text) <= max_len else text[:max_len] + "..."

    df["User Prompt"] = df["User Prompt"].apply(truncate)
    df["Assistant Response"] = df["Assistant Response"].apply(truncate)

    # Display the DataFrame with nicer styling
    display(df.style.set_properties(**{
        'text-align': 'left',
        'white-space': 'pre-wrap'
    }).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]))