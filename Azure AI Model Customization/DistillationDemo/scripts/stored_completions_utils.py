from typing import Optional

import json
import time

import openai
from pathlib import Path

#from tqdm.notebook import tqdm
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed


def process_and_store_completions(
    client: openai.Client,
    input_path: str,
    output_path: str,
    model: str = "o3-mini",
    dataset_name: str = "random",
    author: str = "omkar",
    max_records: int = None,
    system_prompt: str = "You are a helpful assistant.",
    wait_till_stored: bool = False,
):
    """
    Process an input JSONL file containing {"prompt": "xxx"}, call the Azure OpenAI API for stored completions,
    and save the responses to an output file in the format of distilled_output.jsonl.

    Args:
        input_path (str): Path to the input JSONL file containing {"prompt": "xxx"}.
        output_path (str): Path to the output JSONL file.
        dataset_name (str): Metadata for the dataset name. Default is "random".
        author (str): Metadata for the author. Default is "omkar".
        max_records (int): Maximum number of records to process. Default is None (process all records).
        system_prompt (str): System prompt to use for the completions. Default is "You are a helpful assistant."
    """

    input_path = Path(input_path)
    output_path = Path(output_path)

    # Load input data
    with open(input_path, "r", encoding="utf-8") as file:
        data = [json.loads(line) for line in file]

    # Limit the number of records if max_records is specified
    if max_records is not None:
        data = data[:max_records]
    print(f"🔹 Loaded {len(data)} records from {input_path}")

    # Recreate output file if it exists, else create it
    if output_path.exists():
        output_path.unlink()
        print(f"⚠️ Output file {output_path} already exists. Deleting it.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.touch()

    futures = {} # future: prompt
    with ThreadPoolExecutor(thread_name_prefix="chat-completion") as pool:
        for i, entry in enumerate(data):
            prompt = entry.get("prompt")
            if not prompt:
                print(f"⚠️ Skipping record {i}: Missing 'prompt' field.")
                continue
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                "store": True,
                "metadata": {
                    "dataset": dataset_name,
                    "model": model,
                    "index": str(i),
                    "author": author,
                },
            }
            future = pool.submit(client.chat.completions.create, **kwargs)
            futures.update({ future: prompt })

        with open(output_path, "a", encoding="utf-8") as f:
            for future in tqdm(as_completed(futures.keys()), total=len(futures)):
                completion = future.result()
                prompt = futures[future]
                output_record = {
                    "prompt": prompt,
                    "stored_completion_id": completion.id,
                    "preferred_output": completion.choices[0].message.content.strip(),
                }
                # Append the formatted record to the output file
                json.dump(output_record, f)
                f.write("\n")

    print(f"\n✅ Done. Completions saved to: {output_path}")

    if wait_till_stored:
        wait_till_completion_stored(
            client=client,
            expected_count=len(data),
            model=model,
            dataset_name=dataset_name,
            author=author,
        )


def wait_till_completion_stored(
    client: openai.Client,
    expected_count: int,
    model: Optional[str] = None,
    dataset_name: Optional[str] = None,
    author: Optional[str] = None,
):
    metadata = _build_metadata(
        model=model,
        dataset_name=dataset_name,
        author=author,
    )

    while True:
        # List all stored completions
        completions_result = client.chat.completions.list(
            metadata=metadata,
            limit=1,
        )

        if completions_result.total == expected_count:
            break

        print(
            f"Waiting for stored completions to be available... "
            f"Current count: {completions_result.total}, Expected count: {expected_count}"
        )
        time.sleep(5)

    print(f"✅ All {expected_count} stored completions are available.")


def batch_delete_all_stored_completions(
    client: openai.Client,
    model: Optional[str] = None,
    dataset_name: Optional[str] = None,
    author: Optional[str] = None,
):
    metadata = _build_metadata(
        model=model,
        dataset_name=dataset_name,
        author=author,
    )

    completions = []

    after = None
    while True:
        # List all stored completions
        completions_result = client.chat.completions.list(
            metadata=metadata,
            after=after,
            limit=100,
        )
        completions += completions_result.data

        if not completions_result.has_more:
            break

        after = completions_result.last_id

    if not completions:
        print("✅ No stored completions to delete.")

    for completion in tqdm(completions, desc="Deleting stored completions"):
        client.chat.completions.delete(completion.id)

    print(f"✅ Deleted {len(completions)} stored completions.")


def _build_metadata(
    model: Optional[str] = None,
    dataset_name: Optional[str] = None,
    author: Optional[str] = None,
):

    metadata = {}

    if dataset_name:
        metadata["dataset"] = dataset_name
    if model:
        metadata["model"] = model
    if author:
        metadata["author"] = author

    return metadata
