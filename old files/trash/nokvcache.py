import time
import torch
from transformers import AutoTokenizer, OPTForCausalLM
from datasets import load_dataset
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import os

# Paths and parameters
MODEL_NAME = "facebook/opt-350m"
SAVE_DIR = "./results"
TOP_LOW_SENSITIVITY_PERCENT = 30  # Focus on the least sensitive 30% layers
NUM_CLUSTERS = 37  # Number of clusters for attention heads
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SAMPLES = 100  # Limit for dataset size during evaluation

os.makedirs(SAVE_DIR, exist_ok=True)

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = OPTForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)
model.eval()


def preprocess_data(dataset, max_samples=100, max_length=512):
    """
    Preprocess dataset for tokenization.
    """
    inputs = []
    labels = []

    for i, example in enumerate(dataset):
        if i >= max_samples:
            break
        goal = example.get("goal", "")
        sol1 = example.get("sol1", "")
        sol2 = example.get("sol2", "")
        label = example.get("label", None)  # Ground truth label
        text = f"{goal} {sol1} {sol2}"
        inputs.append(text)
        labels.append(label)

    tokenized_inputs = tokenizer(
        inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length
    )
    return tokenized_inputs, labels


def compute_sensitivity_scores(model, tokenized_inputs):
    """
    Compute sensitivity scores for each attention head.
    """
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    sensitivity_scores = torch.zeros(num_layers, num_heads, device=DEVICE)

    input_ids = tokenized_inputs["input_ids"].to(DEVICE)
    attention_mask = tokenized_inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        original_output = model(input_ids, attention_mask=attention_mask).logits

    for layer in tqdm(range(num_layers), desc="Computing sensitivity scores"):
        for head in range(num_heads):
            # Hook to zero out specific attention heads
            def hook_fn(module, input, output):
                output[:, head, :, :] = 0  # Zero out the specific head
                return output

            hook_handle = model.model.decoder.layers[layer].self_attn.out_proj.register_forward_hook(hook_fn)

            # Forward pass with modified attention head
            with torch.no_grad():
                perturbed_output = model(input_ids, attention_mask=attention_mask).logits

            # Compute sensitivity as L2 norm of the difference
            sensitivity_scores[layer, head] = torch.norm(original_output - perturbed_output, p=2)

            hook_handle.remove()

    return sensitivity_scores


def identify_low_sensitivity_layers(sensitivity_scores, percent=TOP_LOW_SENSITIVITY_PERCENT):
    """
    Identify layers with the lowest sensitivity scores.
    """
    layer_scores = sensitivity_scores.mean(dim=1)  # Average sensitivity per layer
    num_layers_to_keep = int(len(layer_scores) * percent / 100)
    low_sensitivity_layers = torch.argsort(layer_scores)[:num_layers_to_keep]
    return low_sensitivity_layers


def extract_attention_keys(model, tokenized_inputs, low_sensitivity_layers):
    """
    Extract attention keys for specified low-sensitivity layers.
    """
    input_ids = tokenized_inputs["input_ids"].to(DEVICE)
    attention_mask = tokenized_inputs["attention_mask"].to(DEVICE)
    keys = []

    def hook_fn(module, input, output):
        keys.append(output.detach().cpu().numpy())  # Extract keys from self-attention

    with torch.no_grad():
        for layer in low_sensitivity_layers:
            hook_handle = model.model.decoder.layers[layer].self_attn.k_proj.register_forward_hook(hook_fn)
            model(input_ids, attention_mask=attention_mask)
            hook_handle.remove()

    return keys


def cluster_attention_heads(keys, num_clusters=NUM_CLUSTERS):
    """
    Cluster attention keys using KMeans.
    """
    flattened_keys = [key.reshape(key.shape[0], -1) for key in keys]
    all_keys = np.concatenate(flattened_keys, axis=0)
    kmeans = KMeans(n_clusters=num_clusters, random_state=42).fit(all_keys)
    return kmeans


def apply_clustering_mask(model, low_sensitivity_layers, cluster_centers, cluster_threshold=5):
    """
    Modify the model to zero out attention heads based on clustering results.
    """
    def cluster_mask_fn(module, input, output):
        # Zero out heads based on cluster proximity
        cluster_mask = (cluster_centers.mean(axis=0) < cluster_threshold).astype(int)
        output[:, cluster_mask == 0, :, :] = 0
        return output

    for layer in low_sensitivity_layers:
        model.model.decoder.layers[layer].self_attn.out_proj.register_forward_hook(cluster_mask_fn)

    return model


def evaluate_model(model, tokenized_inputs, labels):
    """
    Evaluate the model's accuracy and inference time.
    """
    input_ids = tokenized_inputs["input_ids"].to(DEVICE)
    attention_mask = tokenized_inputs["attention_mask"].to(DEVICE)

    start_time = time.time()
    correct = 0

    for i in tqdm(range(len(labels)), desc="Evaluating model"):
        input_id = input_ids[i].unsqueeze(0)  # Add batch dimension
        attention_mask_id = attention_mask[i].unsqueeze(0)

        with torch.no_grad():
            logits = model(input_id, attention_mask=attention_mask_id).logits

        # Extract logits for the final position
        final_token_logits = logits[0, -1, :]
        option1_score = final_token_logits[tokenizer.convert_tokens_to_ids("1")].item()
        option2_score = final_token_logits[tokenizer.convert_tokens_to_ids("2")].item()

        prediction = 0 if option1_score > option2_score else 1
        if prediction == labels[i]:
            correct += 1

    end_time = time.time()
    accuracy = correct / len(labels)
    inference_time = end_time - start_time
    return accuracy, inference_time


def main():
    # Load dataset
    dataset = load_dataset("piqa", split="validation")
    tokenized_inputs, labels = preprocess_data(dataset, max_samples=MAX_SAMPLES)

    # Step 1: Compute sensitivity scores
    print("Computing sensitivity scores...")
    sensitivity_scores = compute_sensitivity_scores(model, tokenized_inputs)

    # Step 2: Identify low-sensitivity layers
    print("Identifying low-sensitivity layers...")
    low_sensitivity_layers = identify_low_sensitivity_layers(sensitivity_scores)

    # Step 3: Extract attention keys from low-sensitivity layers
    print("Extracting attention keys...")
    attention_keys = extract_attention_keys(model, tokenized_inputs, low_sensitivity_layers)

    # Step 4: Cluster attention heads
    print("Clustering attention heads...")
    kmeans = cluster_attention_heads(attention_keys)

    # Evaluate baseline model
    print("Evaluating baseline model...")
    baseline_accuracy, baseline_time = evaluate_model(model, tokenized_inputs, labels)

    # Apply clustering mask to the model
    print("Evaluating clustered model...")
    clustered_model = apply_clustering_mask(model, low_sensitivity_layers, kmeans.cluster_centers_)
    clustered_accuracy, clustered_time = evaluate_model(clustered_model, tokenized_inputs, labels)

    # Compute speedup
    speedup = baseline_time / clustered_time if clustered_time > 0 else 0

    # Save and display results
    print("Saving results...")
    results = {
        "Baseline Accuracy": baseline_accuracy,
        "Baseline Time (s)": baseline_time,
        "Clustered Accuracy": clustered_accuracy,
        "Clustered Time (s)": clustered_time,
        "Speedup": speedup,
    }
    results_path = os.path.join(SAVE_DIR, "evaluation_results.txt")
    with open(results_path, "w") as f:
        for key, value in results.items():
            f.write(f"{key}: {value}\n")
    print("Results saved:", results)


if __name__ == "__main__":
    main()
