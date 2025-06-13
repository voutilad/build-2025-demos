import pandas as pd
import json
from IPython.display import display

def display_sample_examples(dataset, max_len=300):
    """
    Converts a dataset to a Pandas DataFrame and displays sample examples.

    Args:
        dataset: The dataset to process (e.g., evaluation or training dataset).
        max_len: Maximum length for truncating text fields.
    """
    # Convert to Pandas DataFrame
    df = pd.DataFrame(dataset)

    # Select a few samples to display
    sample_df = df[['history', 'human_ref_A', 'human_ref_B', 'labels']].head(3).copy()

    # Map labels: 1 -> 'A', 0 -> 'B'
    sample_df['Preferred'] = sample_df['labels'].apply(lambda x: 'A' if x == 1 else 'B')

    # Rename columns for nicer display
    sample_df = sample_df.rename(columns={
        'history': 'Prompt',
        'human_ref_A': 'Response A',
        'human_ref_B': 'Response B'
    }).drop(columns=['labels'])

    # Shorten long texts for better table fitting
    def shorten(text, max_len=max_len):
        return text if len(text) <= max_len else text[:max_len] + "..."

    sample_df['Prompt'] = sample_df['Prompt'].apply(shorten)
    sample_df['Response A'] = sample_df['Response A'].apply(shorten)
    sample_df['Response B'] = sample_df['Response B'].apply(shorten)

    # Display the table using Pandas
    display(sample_df.style.set_properties(**{
        'text-align': 'left'
    }).set_table_styles([dict(selector='th', props=[('text-align', 'left')])]))

def save_eval_dataset(dataset, output_path_base, include_item_field=False):
    """
    Saves the evaluation dataset in multiple JSONL formats.

    Args:
        dataset: The dataset to save (e.g., evaluation dataset).
        output_path_base: Base path to save the JSONL files (without extension).
        include_item_field: Whether to include the "item" wrapper field in the output.
    """
    # Paths for different output files
    prompts_path = f"{output_path_base}_prompts.jsonl"
    preferred_path = f"{output_path_base}_preferred.jsonl"
    rejected_path = f"{output_path_base}_rejected.jsonl"
    full_path = f"{output_path_base}_full.jsonl"

    # Open all files for writing
    with open(prompts_path, 'w', encoding='utf-8') as prompts_file, \
         open(preferred_path, 'w', encoding='utf-8') as preferred_file, \
         open(rejected_path, 'w', encoding='utf-8') as rejected_file, \
         open(full_path, 'w', encoding='utf-8') as full_file:

        for _, row in enumerate(dataset):
            # Determine preferred and rejected responses based on labels
            preferred = row['human_ref_A'] if row['labels'] == 1 else row['human_ref_B']
            rejected = row['human_ref_B'] if row['labels'] == 1 else row['human_ref_A']

            # Create the record structures
            prompt_record = {"input": row['history']}
            preferred_record = {"input": row['history'], "output": preferred}
            rejected_record = {"input": row['history'], "output": rejected}

            full_record = {
                "prompt": row['history'],
                "preferred": preferred,
                "rejected": rejected
            }

            # Wrap in "item" field if include_item_field is True
            if include_item_field:
                full_record = {"item": full_record}

            # Write to respective files
            prompts_file.write(json.dumps(prompt_record) + '\n')
            preferred_file.write(json.dumps(preferred_record) + '\n')
            rejected_file.write(json.dumps(rejected_record) + '\n')
            full_file.write(json.dumps(full_record) + '\n')

    # Print confirmation messages
    print(f"✅ Prompts-only dataset saved to {prompts_path}")
    print(f"✅ Preferred completions dataset saved to {preferred_path}")
    print(f"✅ Rejected completions dataset saved to {rejected_path}")
    print(f"✅ Full dataset saved to {full_path}")

def save_train_dataset(dataset, output_path):
    """
    Saves the training dataset in JSONL format.

    Args:
        dataset: The dataset to save (e.g., training dataset).
        output_path: Path to save the JSONL file.
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in dataset:
            record = {
                "prompt": row['history'],
                "completion": row['human_ref_A'] if row['labels'] == 1 else row['human_ref_B']
            }
            f.write(json.dumps(record) + '\n')
    print(f"✅ Training dataset saved to {output_path}")