from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import pandas as pd
import torch
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset
import time
import re


def code_from_ans(result):
    match = re.findall(r"\[[0-9, ]+\]", result)
    if match:
        return match[0].strip('[]')
        
def run_ds(input_data, pre_note_message, message_template, output_csv_path, text_column='NARRATIVE', true_code_col='binarySzType', num_rows=None, seed=42, code_from_ans_func = code_from_ans, temp = 1, max_token=100):
    """
    Runs a text generation pipeline using a pre-trained language model to process clinical notes 
    and generate predictions based on a provided message template.

    Args:
        input_data (pd.DataFrame): A pandas DataFrame containing the input data. 
            Must include a column specified by `text_column` and optionally 'filename', 'szType', and 'binarySzType'.
        message_template (str): A string template to append to each clinical note for generating predictions.
        output_csv_path (str): The file path where the generated predictions will be saved as a CSV file.
        text_column (str, optional): The name of the column in `input_data` containing the text to process. 
            Defaults to 'text'.
        num_rows (int, optional): The number of rows to process from `input_data`. If None, all rows are processed. 
            Defaults to None.

    Returns:
        None: The function saves the generated predictions to the specified CSV file.

    Notes:
        - The function uses the Hugging Face Transformers library to load a pre-trained model and tokenizer.
        - The input data is converted into a Hugging Face Dataset for efficient processing.
        - Predictions are generated using a text-generation pipeline and saved along with metadata 
        (e.g., filename, original text, true labels) to a CSV file.
        - The function processes the data in batches and logs progress every 10 notes.
    """
    # Start tracking runtime
    start_time = time.time()
    torch.manual_seed(seed)
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    deepseek_path = '/models/DeepSeek-R1-Distill-Llama-8B'
    model = AutoModelForCausalLM.from_pretrained(deepseek_path, device_map='auto', torch_dtype=torch.float16)
    tokenizer = AutoTokenizer.from_pretrained(deepseek_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, trust_remote_code=False, device_map='auto')

    
    if num_rows is not None:
        input_data = input_data.head(num_rows)
    section_search_messages = [
        {"content": f"{pre_note_message} '{row[text_column]}' {message_template}"}
        for _, row in input_data.iterrows()
    ]
    message_dataset = Dataset.from_list(section_search_messages)

    # Generate predictions
    print("Generating predictions...")
    generated_texts = []
    with torch.no_grad():
        predictions = pipe(KeyDataset(message_dataset, "content"), max_new_tokens=max_token, return_full_text=False, batch_size=8)
        for idx, output in enumerate(predictions):
            generated_texts.append({
                "prediction": code_from_ans_func(output[0]['generated_text'].split('\n</think>\n\n')[-1],output[0]['generated_text']),
                "full_output": output[0]['generated_text'],
                "original_text": input_data.iloc[idx][text_column],
            })
            torch.cuda.empty_cache()
            if idx % 10 == 0:
                print(f"Processed {idx} notes...")

                # Save results
                print(f"Saving results to {output_csv_path}...")
                results_df = pd.DataFrame(generated_texts)
                results_df.to_csv(output_csv_path, index=False)
                print("Results saved.")
    
    results_df = pd.DataFrame(generated_texts)
    results_df.to_csv(output_csv_path, index=False)
    # End tracking runtime
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total runtime: {elapsed_time:.2f} seconds")