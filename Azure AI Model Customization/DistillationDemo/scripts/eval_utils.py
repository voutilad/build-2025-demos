import requests
import asyncio
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  # Import numpy for percentile calculations

from dotenv import load_dotenv
import os

# Load environment variables from the .env file
load_dotenv()

# API keys and endpoint
OAI_API_TYPE = os.getenv("OAI_API_TYPE", "azure").lower()
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_API_ENDPOINT = os.getenv("AZURE_API_ENDPOINT") + "/openai"
API_VERSION = os.getenv("API_VERSION", "2025-04-01-preview")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")


def get_api_config():
    """
    Get the appropriate API configuration based on the OAI_API_TYPE environment variable.

    Returns:
        dict: A dictionary containing the base URL and headers for the API.
    """
    if OAI_API_TYPE == "openai":
        return {
            "base_url": OPENAI_API_BASE + "/v1",
            "headers": {"Authorization": f"Bearer {OPENAI_API_KEY}"},
        }
    else:
        return {
            "base_url": AZURE_API_ENDPOINT,
            "headers": {"api-key": AZURE_API_KEY},
        }


async def create_eval(pass_threshold: float, grader_model: str, name: str, with_sample: bool = True):
    """
    Create an evaluation with the given parameters.

    Args:
        pass_threshold (float): The pass threshold for the evaluation.
        grader_model (str): The grader model to use.
        name (str): The name of the evaluation.

    Returns:
        str: The ID of the created evaluation, or None if creation failed.
    """
    api_config = get_api_config()
    response_placeholder = "sample.output_text" if with_sample else "item.output"
    response = await asyncio.to_thread(
        requests.post,
        f'{api_config["base_url"]}/evals',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
        json={
            'name': name,
            'data_source_config': {
                "type": "custom",
                "include_sample_schema": with_sample,
                "item_schema": {
                    "type": "object",
                    "properties": {
                        "input": {"type": "string"},
                        "output": {"type": "string"},
                    }
                }
            },
            'testing_criteria': [
                {
                    "name": "Auto grader",
                    "type": "score_model",
                    "model": grader_model,
                    "input": [
                        {
                            "type": "message",
                            "role": "developer",
                            "content": {
                                "type": "input_text",
                                "text": "We've created a consumer-facing Evals product to help AI integrators quickly and clearly understand their models' real-world performance. Your role is to serve as a Universal Evaluator, automatically grading responses to measure how well each model output addresses user needs and expectations.\n\nGiven the conversation messages, assign a quality score in the `result` key of the response in the inclusive range between 1.0 (poor) and 7.0 (excellent). Customers will analyze your collective scores and reasoning to gain actionable insights into their models' performance.\n\n---\n\n## Things to Consider\n\n- Evaluate the overall value provided to the user\n-  Verify all claims and do not take the AI's statements at face value! Errors might be very hard to find and well hidden.\n- Differentiate between minor errors (slight utility reduction) and major errors (significant trust or safety impact).\n- Reward answers that closely follow user instructions.\n- Reserve the highest and lowest reward scores for cases where you have complete certainty about correctness and utility.\n\n\n---\n\n## Secondary Labels to Support Final Utility Score Prediction\n\nTo help you assign an accurate final utility score, first analyze and predict several important aspects of the AI response. Crucially, these intermediate evaluations should precede your final utility score prediction.\n\nYour structured output must match the provided schema:\n\n- `steps`: A JSON array of objects, each containing:\n  - `description`: A detailed explanation of your reasoning for each step.\n  - `result`: The float score reached based on the reasoning in this step.\n\n### Steps to Predict (in order):\n\n1. **major_errors**\n   - *description*: Identify and explain any significant errors.\n   - *conclusion*: List major errors found, or indicate \"None\".\n\n2. **minor_errors**\n   - *description*: Identify and explain any minor inaccuracies.\n   - *conclusion*: List minor errors found, or indicate \"None\".\n\n3. **potential_improvements**\n   - *description*: Suggest enhancements that would improve the response.\n   - *conclusion*: List suggested improvements, or indicate \"None\".\n\n---\n\n## JSON Response Structure\n\nOnce you predicted all the above fields you need to assign a float between 1 and 7 to indicate the response's utility compared to the alternative responses. Use your best judgment for the meaning of `final_score`. Your response should be a JSON that can be loaded with json.loads in Python and contains:\n- steps: An array of objects representing your reasoning steps. Each step includes:\n  - description (string): Detailed reasoning for this step.\n  - result (string): The float score derived from this reasoning.\n- result (float): A numeric quality score as a string, in the inclusive range [1,7].\n\n---\n\n## Notes\n\n- Be meticulous in identifying errors, especially subtle or high-impact ones.\n- Avoid being too kind by giving overly high scores easily, it's important to often keep a gap at the top to continue having signal for improvement. Only use [6.5, 7) if the answer is truly mind blowing and you don't see how it could have been improved.\n- Never take the AI's responses at face value—verify everything thoroughly.\n"
                            }
                        },
                        {
                            "type": "message",
                            "role": "user",
                            "content": {
                                "type": "input_text",
                                "text": f"**User input**\n\n{{{{item.input}}}}\n\n**Response to evaluate**\n\n{{{{{response_placeholder}}}}}\n\n"
                            }
                        }
                    ],
                    "pass_threshold": pass_threshold,
                    "range": [1.0, 7.0]
                }
            ]
        }
    )

    if response.status_code in (200, 201):
        eval_id = response.json().get('id')
        print(f"Evaluation created successfully with ID: {eval_id}")
        return eval_id
    else:
        print(f"Failed to create evaluation. Status Code: {response.status_code}")
        print(response.text)
        return None


async def create_eval_run(eval_id, file_id, model_deployment=None, eval_run_name=None, use_sample=True):
    """
    Create an evaluation run for a specific model deployment or a custom evaluation run.

    Args:
        eval_id (str): The evaluation ID.
        file_id (str): The file ID to use for the evaluation.
        model_deployment (str, optional): The model deployment name. Required if `use_sample` is True.
        eval_run_name (str, optional): The evaluation run name. Required if `use_sample` is False.
        use_sample (bool, optional): Whether to use the sample-based evaluation. Defaults to True.

    Returns:
        None
    """
    api_config = get_api_config()

    # Construct the JSON payload based on the `use_sample` flag
    if use_sample:
        if not model_deployment:
            raise ValueError("`model_deployment` is required when `use_sample` is True.")
        payload = {
            "name": f"AutoGrader_{model_deployment}",
            "data_source": {
                "type": "completions",
                "source": {"type": "file_id", "id": file_id},
                "input_messages": {
                    "type": "template",
                    "template": [
                        {"type": "message", "role": "developer", "content": {"type": "input_text", "text": "You are a helpful assistant, answer the queries"}},
                        {"type": "message", "role": "user", "content": {"type": "input_text", "text": "{{item.input}}"}}
                    ]
                },
                "model": model_deployment,
            },
        }
    else:
        if not eval_run_name:
            raise ValueError("`eval_run_name` is required when `use_sample` is False.")
        payload = {
            "name": eval_run_name,
            "data_source": {
                "type": "jsonl",
                "source": {"type": "file_id", "id": file_id}
            },
        }

    # Make the API request
    response = await asyncio.to_thread(
        requests.post,
        f'{api_config["base_url"]}/evals/{eval_id}/runs',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
        json=payload)

    # print response code and error message if any
    if response.status_code not in (200, 201):
        print(f"Failed to create evaluation run. Status Code: {response.status_code}")
        print(response.text)
        return None
    

    # Log the response status
    if use_sample:
        print(f"Create Eval Run Status for {model_deployment}: {response.status_code}")
    else:
        print(f"Create Eval Run Status for {eval_run_name}: {response.status_code}")


# Main function to handle multiple deployments
async def run_evaluations(eval_id, file_id, deployments):
    for deployment in deployments:
        await create_eval_run(eval_id, file_id, model_deployment=deployment)
    print(f"Evaluation ID: {eval_id}")


async def get_eval_runs_list(eval_id: str) -> list:
    """
    Fetch the list of evaluation runs for a given evaluation ID.

    Args:
        eval_id (str): The evaluation ID.

    Returns:
        list: A list of evaluation runs with their details.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals/{eval_id}/runs',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    print(f"Get Evaluation Runs: {eval_id}")
    content = response.json().get('data', None)
    list_runs = []

    if content:
        for run in content:
            r = {
                'id': run.get('id', None),
                'name': run.get('name', None),
                'status': run.get('status', None),
                'model': run.get('model', None),
            }
            result = run.get('result_counts', None)
            if result:
                passed = result.get('passed', 0)
                errored = result.get('errored', 0)
                failed = result.get('failed', 0)
                total = result.get('total', 0)
                pass_percentage = (passed * 100) / (passed + failed) if total > 0 else 0
                error_percentage = (errored * 100) / total if total > 0 else 0
                r['pass_percentage'] = pass_percentage
                r['error_percentage'] = error_percentage

            list_runs.append(r)

    return list_runs


async def get_eval_details(eval_id: str) -> dict:
    """
    Fetch the details of a specific evaluation.

    Args:
        eval_id (str): The evaluation ID.

    Returns:
        dict: A dictionary containing evaluation details, including the name.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals/{eval_id}',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code == 200:
        return response.json()
    else:
        print(f"Failed to fetch evaluation details for ID: {eval_id}. Status Code: {response.status_code}")
        return {"name": f"Unknown Evaluation ({eval_id})"}


async def display_evaluation_summary(eval_ids: list):
    """
    Fetch and display a summary of evaluation runs for a list of evaluation IDs, including a horizontal bar chart,
    average score, and score distribution for all runs in a single chart with a maximum of 4 graphs per row.

    Args:
        eval_ids (list): A list of evaluation IDs.
    """
    all_eval_runs = []
    eval_id_to_name = {}
    eval_id_to_color = {}

    # Assign unique colors for each evaluation ID
    colors = plt.cm.tab10.colors  # Use a colormap for distinct colors
    for i, eval_id in enumerate(eval_ids):
        eval_id_to_color[eval_id] = colors[i % len(colors)]

    # Fetch evaluation runs and details for each evaluation ID
    for eval_id in eval_ids:
        eval_runs = await get_eval_runs_list(eval_id)

        # Fetch evaluation details using the helper method
        eval_details = await get_eval_details(eval_id)
        eval_name = eval_details.get('name', f'Unknown Evaluation ({eval_id})')
        eval_id_to_name[eval_id] = eval_name

        # Add evaluation ID to each run for color coding
        for run in eval_runs:
            run['eval_id'] = eval_id
            all_eval_runs.append(run)

    # Combine all evaluation runs into a single DataFrame
    if all_eval_runs:
        df = pd.DataFrame(all_eval_runs)
        df = df[['id', 'name', 'model', 'status', 'pass_percentage', 'error_percentage', 'eval_id']]  # Select relevant columns
        df['eval_name'] = df['eval_id'].map(eval_id_to_name)  # Map eval_id to eval_name
        df['model'] = df['model'].str[:15]  # Truncate model names to 15 characters
        df = df.sort_values(by=['pass_percentage'], ascending=[False])  # Sort by pass_percentage descending

        print("\n" + "=" * 50)
        print("Combined Evaluation Summary")
        print("=" * 50)
        print(df.to_string(index=False, header=["Run ID", "Run Name", "Model", "Status", "Pass Percentage (%)", "Error Percentage (%)", "Evaluation ID", "Evaluation Name"]))
        print("=" * 50)

        # Dynamically adjust the figure height based on the number of rows
        num_rows = len(df)
        fig_height = max(3, num_rows * 0.5)  # Set a minimum height of 6 and scale with 0.5 per row


        # Create a horizontal bar chart with rows sorted by pass percentage across all eval_ids
        plt.figure(figsize=(12, fig_height))

        df['display_label'] = df['model'].where(
            (df['model'].str.strip() != '') & (df['model'] != 'None') & (df['model'].notna()),
            df['name']
            )
        
        plt.barh(
            df['display_label'], 
            df['pass_percentage'], 
            color=[eval_id_to_color[eval_id] for eval_id in df['eval_id']], 
            edgecolor='black'
        )
        plt.xlabel('Pass Percentage (%)')
        plt.ylabel('Model')
        plt.title("Pass Percentage by Model Across Evaluations")
        plt.xlim(0, 100)  # Set x-axis scale explicitly to 0-100
        plt.gca().invert_yaxis()  # Invert y-axis to show the highest percentage at the top
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()

        # Process each run to calculate and collect scores for distribution
        all_scores = []
        run_labels = []
        score_summary = []  # To store data for the summary table
        for _, row in df.iterrows():
            run_id = row['id']
            model = row['model']
            eval_id = row['eval_id']
            scores = await get_eval_run_output_items(eval_id, run_id)

            if scores:
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                p10 = np.percentile(scores, 10)  # 10th percentile
                p25 = np.percentile(scores, 25)  # 25th percentile
                p50 = np.percentile(scores, 50)  # 50th percentile (median)
                p75 = np.percentile(scores, 75)  # 75th percentile
                p90 = np.percentile(scores, 90)  # 90th percentile

                # Collect scores and labels for the combined chart
                all_scores.append((scores, eval_id_to_color[eval_id]))  # Include color for the subplot
                run_labels.append(f"{model} ({eval_id_to_name[eval_id]})")  # Include eval name in the label

                # Add data to the summary table
                score_summary.append({
                    "Model": model,
                    "Evaluation Name": eval_id_to_name[eval_id],
                    "Average Score": f"{avg_score:.2f}",
                    "Min Score": f"{min_score:.2f}",
                    "Max Score": f"{max_score:.2f}",
                    "10th Percentile": f"{p10:.2f}",
                    "25th Percentile": f"{p25:.2f}",
                    "50th Percentile": f"{p50:.2f}",
                    "75th Percentile": f"{p75:.2f}",
                    "90th Percentile": f"{p90:.2f}"
                })

        # Display the score summary as a table
        if score_summary:
            score_df = pd.DataFrame(score_summary)
            score_df = score_df.sort_values(by=['Evaluation Name', 'Average Score'], ascending=[True, False])  # Sort by eval_name and avg_score
            print("\n" + "=" * 50)
            print("Score Summary Table:")
            print(score_df.to_string(index=False))
            print("=" * 50)

        # Plot all score distributions in a single chart with a maximum of 4 graphs per row
        if all_scores:
            num_runs = len(all_scores)
            max_cols = 4  # Maximum number of graphs per row
            num_rows = (num_runs + max_cols - 1) // max_cols  # Calculate the number of rows

            fig, axes = plt.subplots(num_rows, max_cols, figsize=(5 * max_cols, 4 * num_rows), sharey=True)
            axes = axes.flatten()  # Flatten the axes array for easier indexing

            for i, ((scores, color), label) in enumerate(zip(all_scores, run_labels)):
                ax = axes[i]
                ax.hist(scores, bins=10, color=color, edgecolor='black')  # Use color for the histogram
                ax.set_title(label, fontsize=10)  # Include model and evaluation name
                ax.set_xlabel("Score")
                ax.set_ylabel("Frequency")
                ax.set_xlim(0, 7)  # Fix the x-axis range between 0 and 7
                ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Hide any unused subplots
            for j in range(len(all_scores), len(axes)):
                axes[j].axis('off')

            plt.tight_layout()
            plt.suptitle("Score Distributions for each Model", fontsize=16, y=1.02)
            plt.show()
    else:
        print("\n" + "=" * 50)
        print("No evaluation runs found for the provided Evaluation IDs.")
        print("=" * 50)


async def list_evaluations() -> list:
    """
    Fetch the list of all evaluations.

    Returns:
        list: A list of evaluations with their details.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code == 200:
        content = response.json().get('data', [])
        print("Fetched evaluations successfully.")
        return content
    else:
        print(f"Failed to fetch evaluations. Status Code: {response.status_code}")
        return []


async def delete_evaluation(eval_id: str) -> bool:
    """
    Delete an evaluation by its ID.

    Args:
        eval_id (str): The evaluation ID to delete.

    Returns:
        bool: True if the evaluation was deleted successfully, False otherwise.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.delete,
        f'{api_config["base_url"]}/evals/{eval_id}',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code == 204:
        print(f"Evaluation with ID {eval_id} deleted successfully.")
        return True
    else:
        print(f"Failed to delete evaluation with ID: {eval_id}. Status Code: {response.status_code}")
        return False


async def delete_all_evaluations():
    """
    Delete all evaluations.

    Returns:
        None
    """
    # Fetch all evaluations
    evaluations = await list_evaluations()

    if not evaluations:
        print("No evaluations found to delete.")
        return

    # Iterate through each evaluation and delete it
    for evaluation in evaluations:
        eval_id = evaluation.get('id')
        if eval_id:
            success = await delete_evaluation(eval_id)
            if success:
                print(f"Deleted evaluation with ID: {eval_id}")
            else:
                print(f"Failed to delete evaluation with ID: {eval_id}")


async def get_eval_run_output_items(eval_id: str, run_id: str) -> list:
    """
    Fetch the output items for a specific evaluation run and extract the result scores.

    Args:
        eval_id (str): The evaluation ID.
        run_id (str): The run ID.

    Returns:
        list: A list of scores for the output items.
    """
    api_config = get_api_config()
    response = await asyncio.to_thread(
        requests.get,
        f'{api_config["base_url"]}/evals/{eval_id}/runs/{run_id}/output_items',
        params={'api-version': API_VERSION} if OAI_API_TYPE == "azure" else None,
        headers=api_config["headers"],
    )

    if response.status_code == 200:
        output_items = response.json().get('data', [])
        scores = []
        for item in output_items:
            results = item.get('results', [])
            for result in results:
                score = result.get('score')
                if score is not None:
                    scores.append(score)
        return scores
    else:
        print(f"Failed to fetch output items for run {run_id}. Status Code: {response.status_code}")
        return []