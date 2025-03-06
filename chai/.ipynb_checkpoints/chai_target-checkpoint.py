import torch
from transformers import AutoConfig, Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import copy
import numpy as np

def compute_sensitivity(attention_scores):
    # Debugging: Print attention scores
    print("Raw Attention Scores:", attention_scores)

    # Compute absolute mean for each layer
    cleaned_scores = {
        layer_idx: np.mean(np.abs(np.array(scores, dtype=np.float32)))
        for layer_idx, scores in attention_scores.items()
        if isinstance(scores, (list, np.ndarray)) and len(scores) > 0  # Ensure non-empty values
    }
    
    # Debugging: Print computed sensitivities
    print("Computed Sensitivities:", cleaned_scores)
    
    return cleaned_scores
def get_attention_scores(model, input_ids):
    """Extracts attention scores from the model while ensuring correct dimensions."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_ids = input_ids.to(device)  #  Move input IDs to GPU
    model.to(device)
    
    attention_scores = {}

    with torch.no_grad():
        outputs = model(input_ids)

        if hasattr(outputs, "attentions") and outputs.attentions is not None:
            for layer_idx, attn in enumerate(outputs.attentions):
                attn = attn.cpu().numpy()  #  Move to CPU
                if attn.ndim == 4:  # Expected shape: (batch_size, num_heads, seq_length, seq_length)
                    attn = np.mean(attn, axis=(0, 2, 3))  #  Average across heads and sequences
                elif attn.ndim == 3:  # Unexpected case, still averaging
                    attn = np.mean(attn, axis=(0, 2))
                elif attn.ndim == 2:  # More unexpected cases
                    attn = np.mean(attn, axis=0)
                
                attention_scores[layer_idx] = attn  #  Store correctly processed attention scores

        else:
            return {"error": "Model does not return attention scores. Check model architecture."}

    return attention_scores
def main_chai_target(model, tokenizer, dataset_name, epochs=30, batch_size=16, learning_rate=2e-5):
    """
    Identifies and perturbs the most sensitive layers of the model based on accuracy drop.
    Returns the modified model with targeted layers fine-tuned.

    Args:
        model (torch.nn.Module): The pre-trained model to analyze.
        tokenizer: Tokenizer associated with the model.
        dataset_name (str): Name of the dataset to use ('rte', 'piqa', or 'sst2').
        epochs (int, optional): Number of fine-tuning epochs. Defaults to 3.
        batch_size (int, optional): Batch size for training. Defaults to 16.
        learning_rate (float, optional): Learning rate for fine-tuning. Defaults to 2e-5.

    Returns:
        torch.nn.Module: The model after targeted fine-tuning on sensitive layers.
    """
    print("🔹 [chai-target] Before modification: model parameters")
    for name, param in model.named_parameters():
      print(f"  {name}: {param.shape}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
# If model is in half precision and running on CPU, convert to float32
    if device.type == "cpu":
        model = model.float()  #  Converts model weights to float32 for CPU compatibility

    # Mapping dataset names to Hugging Face's dataset format
    dataset_mapping = {
        "sst2": ("glue", "sst2"),
        "rte": ("glue", "rte"),
        "piqa": ("piqa", "validation"),
    }

    if dataset_name not in dataset_mapping:
        raise ValueError(f"Dataset '{dataset_name}' is not supported.")

    dataset_source, dataset_subset = dataset_mapping[dataset_name]
    dataset = load_dataset(dataset_source, dataset_subset, split="train")

    # Determine the appropriate text column
    text_column = 'sentence' if 'sentence' in dataset.column_names else 'goal'

    # Preprocessing function for tokenization
    def preprocess_function(examples):
        return tokenizer(examples[text_column], truncation=True, padding='max_length', max_length=512)  

    # Tokenize the dataset
    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True)

    # Identify the most sensitive layers
    sensitivities = []

    # Placeholder: Manually selected sensitive layers
    input_ids = torch.randint(0, 50256, (1, 32))
    attention_scores = get_attention_scores(model, input_ids)
    sensitivities = compute_sensitivity(attention_scores)
# Compute the threshold for top 30% (Only if sensitivities are not empty)
    if sensitivities:
        num_layers = len(sensitivities)

        # Compute the threshold for top 30%
        top_k = int(0.3 * num_layers)

        # Get the indices of the top 30% most sensitive layers
        top_layer_indices = torch.argsort(sensitivities, descending=True)[:top_k]
        targeted_layers= top_layer_indices.tolist()
        print(f"Identified Targeted Layers: {targeted_layers}")
    else:
        print(" Warning: No sensitivities detected. Skipping targeted fine-tuning.")
        targeted_layers = [2,3,4]  #  Skip fine-tuning if no layers are detected

    print(f"Identified Targeted Layers: {targeted_layers}")


    #  **Fine-Tune Only Targeted Layers**
    print("Starting fine-tuning on targeted layers...")

    # Freeze all layers except the targeted layers
    for layer_num, layer in enumerate(model.model.decoder.layers):
        for param in layer.parameters():
            param.requires_grad = layer_num in targeted_layers  # Only fine-tune targeted layers

        # Define training arguments
    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        evaluation_strategy="epoch" if dataset_name != "piqa" else "no",  #  Fix for missing eval dataset
        save_strategy="epoch",
        learning_rate=learning_rate,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,  #  Ensuring evaluation dataset is provided
    )

    # Fine-tune the model
    trainer.train()



    # Fine-tune the model
    trainer.train()

    # Ensure model is correctly returned
    if hasattr(trainer.model, "module"):
        return trainer.model.module  # Extract if wrapped in DataParallel
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape}")

    return trainer.model  # Returning the fine-tuned model
