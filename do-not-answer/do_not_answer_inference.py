import json
import torch
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation.utils import GenerationConfig

# Load model and tokenizer
def load_model_and_tokenizer():
    try:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", use_fast=False, trust_remote_code=True)
        print("Tokenizer loaded")
        
        print("Loading model...")
        model = AutoModelForCausalLM.from_pretrained("baichuan-inc/Baichuan2-7B-Chat", device_map="cuda", torch_dtype=torch.bfloat16, trust_remote_code=True)
        print("Model loaded")
        
        print("Setting generation config...")
        model.generation_config = GenerationConfig.from_pretrained("baichuan-inc/Baichuan2-7B-Chat")
        print("Generation config set")
        
        return model, tokenizer
    except Exception as e:
        print(f"Error during model/tokenizer loading: {e}")
        return None, None

# Get response from the model
def get_model_response(model, tokenizer, message):
    try:
        print(f"Running model for prompt: {message}")
        response = model.chat(tokenizer, [{"role": "user", "content": message}])
        print("Response received")
        return response
    except Exception as e:
        print(f"Error during model inference: {e}")
        return None

# Process CSV and generate responses
def process_csv(input_file, output_file, batch_size=15):
    model, tokenizer = load_model_and_tokenizer()
    if model is None or tokenizer is None:
        print("Model or tokenizer failed to load.")
        return
    
    responses = []
    count = 0  # Track the number of processed prompts
    
    try:
        df = pd.read_csv(input_file)
        
        for _, row in df.iterrows():
            prompt = row.get("question", "")
            print(f"Processing prompt {count+1}: {prompt}")
            
            response = get_model_response(model, tokenizer, prompt)
            
            responses.append({
                "id": row["id"],
                "risk_area": row["risk_area"],
                "types_of_harm": row["types_of_harm"],
                "specific_harms": row["specific_harms"],
                "question": prompt,
                "response": response if response else "Error: No response from model."
            })
            
            count += 1
            
            # Save every `batch_size` responses
            if count % batch_size == 0:
                save_to_file(responses, output_file)
                responses = []  # Clear memory
                print(f"Saved {count} responses so far.")
        
        # Save any remaining responses after loop ends
        if responses:
            save_to_file(responses, output_file)
            print(f"Final save: {count} responses processed.")
    
    except Exception as e:
        print(f"Error processing the CSV file: {e}")

# Function to append responses to file
def save_to_file(responses, output_file):
    try:
        with open(output_file, 'a', encoding='utf-8') as f:
            for response in responses:
                f.write(json.dumps(response, ensure_ascii=False) + '\n')  # Append line by line
        print(f"Saved {len(responses)} responses to {output_file}")
    except Exception as e:
        print(f"Error saving responses to output file: {e}")

# Example usage
if __name__ == "__main__":
    input_file = "ROSE/datasets/do-not-answer/do_not_answer_en.csv"
    output_file = "responses_do_not_answer.jsonl"
    process_csv(input_file, output_file)
