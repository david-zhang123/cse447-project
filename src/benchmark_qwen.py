from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
from tqdm import tqdm
from pathlib import Path

# Set seed for reproducibility
random.seed(0)

# Set the local directory for saving the model
LOCAL_MODEL_DIR = "local_qwen_model"

def query_qwen_model(input_text, model_name="Qwen/Qwen3-0.6B"):
    """Query the Qwen model to generate exactly 3 characters."""
    local_model_path = Path(LOCAL_MODEL_DIR)

    # Check if the model is already saved locally
    if not local_model_path.exists():
        print(f"Saving model locally to {LOCAL_MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.save_pretrained(LOCAL_MODEL_DIR)
        model.save_pretrained(LOCAL_MODEL_DIR)
    else:
        print(f"Loading model from local directory {LOCAL_MODEL_DIR}...")
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_DIR)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_DIR)

    # Tokenize the input text
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate output with a maximum of 3 tokens
    outputs = model.generate(
        inputs["input_ids"],
        max_new_tokens=3,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.7
    )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text


if __name__ == "__main__":
    lowercase = True

    test_data = list(load_dataset("papluca/language-identification", split="test")["text"])  # Convert to list
    correct_next_char = []
    predictions = []

    for i in tqdm(range(len(test_data))):
        # Convert to lowercase if toggle is enabled
        test_data[i] = test_data[i].strip()
        if lowercase:
            test_data[i] = test_data[i].lower()
        if len(test_data[i]) < 2:
            continue
        index = random.randint(1, len(test_data[i]) - 1)
        # next character is correct_next_char
        correct_next_char.append(test_data[i][index]) 
        # strip context to right before correct next char
        test_data[i] = test_data[i][:index]

        # Query the Qwen model for predictions
        prediction = query_qwen_model(test_data[i])
        predictions.append(prediction)

    # Write predictions to pred.txt
    with open('pred.txt', 'wt') as f:
        for p in predictions:
            f.write(f"{p}\n")

    # Write correct next char to file for evaluation
    with open('output/correct_next_char.txt', 'wt') as f:
        for c in correct_next_char:
            f.write(f"{c}\n")