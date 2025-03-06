from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import OPTForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.cluster import KMeans
from kneed import KneeLocator
import time
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
                attn = attn.cpu().numpy()  # Move to CPU
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

def divide_layers_by_sensitivity(sensitivities):
    """ Splits layers into 3 groups (High, Medium, Low) based on sensitivity scores. """
    sorted_layers = sorted(sensitivities, key=sensitivities.get, reverse=True)
    num_layers = len(sorted_layers)
    high = sorted_layers[: int(num_layers * 0.2)]
    medium = sorted_layers[int(num_layers * 0.2) : int(num_layers * 0.7)]
    low = sorted_layers[int(num_layers * 0.7) :]
    return high, medium, low

def apply_mixed_precision(model, medium, low):
    """ Applies mixed precision quantization to the model. """
    for layer_idx in medium + low:
        for param in model.model.decoder.layers[layer_idx].parameters():
            param.data = param.data.half()
    model.half()
    return model

def compute_sensitivity(attention_scores):
    cleaned_scores = {
        layer_idx: np.mean(np.abs(np.array(scores, dtype=np.float32)))
        for layer_idx, scores in attention_scores.items()
        if isinstance(scores, (list, np.ndarray))
    }
    return cleaned_scores

# -------------------- Evaluation -------------------- #

def chai_quant_enhancement(chai_base_model,tokenizer,dataset_name):
    #  Save & Reload to Apply Pruning
    #  Measure Model Size After Pruning
    input_ids = torch.randint(0, 50256, (1, 32))
    attention_scores = get_attention_scores(chai_base_model, input_ids)
    sensitivities = compute_sensitivity(attention_scores)
    high, medium, low = divide_layers_by_sensitivity(sensitivities)

    #  Apply Mixed Precision Quantization (CHAI-Quant)
    print("\n Applying Mixed Precision Quantization (CHAI-Quant)...")
    chai_quant_model = apply_mixed_precision(chai_base_model, medium, low)
    print(" [chai-quant] After modification: model parameters")
    for name, param in chai_quant_model.named_parameters():
        print(f"  {name}: {param.shape}")

    return chai_quant_model
