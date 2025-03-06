from flask import Flask, request, jsonify, render_template
import torch
import torch.nn.functional as F
import numpy as np
import os
from transformers import AutoConfig, AutoModelForSequenceClassification, OPTForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset
from sklearn.cluster import KMeans
from kneed import KneeLocator
import time
from chai_quant import chai_quant_enhancement
from chai_kd import chai_knowledge_distillation_enhancement
from chai_target import main_chai_target
from original_chai import apply_pruning
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import torch
from tqdm import tqdm

app = Flask(__name__)
def get_model_size(model, path="temp_model.pth"):
    """ Saves model temporarily and checks disk size. """
    torch.save(model.state_dict(), path)
    size_mb = os.path.getsize(path) / (1024 * 1024)  # Convert bytes to MB
    # os.remove(path)  #  Clean up after measurement
    return size_mb



def evaluate_model(model, tokenizer, dataset_name):
    """Evaluates model accuracy on the given dataset."""

    #  Dataset mappings for correct loading
    dataset_mapping = {
        "sst2": ("glue", "sst2", "validation", "sentence"),
        "rte": ("glue", "rte", "validation", ("sentence1", "sentence2")),  # RTE has two sentences
        "piqa": ("piqa", None, "validation", ("goal", "sol1", "sol2")),  # PIQA has different format
    }

    if dataset_name not in dataset_mapping:
        return {"error": f"Unsupported dataset: {dataset_name}"}

    dataset_source, dataset_subset, split_name, text_key = dataset_mapping[dataset_name]

    #  Load dataset correctly (PIQA has no `dataset_subset`)
    try:
        if dataset_subset:
            dataset = load_dataset(dataset_source, dataset_subset, split=split_name)
        else:  # PIQA case
            dataset = load_dataset("piqa", split="validation", cache_dir="./cache", download_mode="force_redownload")

    except Exception as e:
        return {"error": f"Error loading dataset: {str(e)}"}

    dataloader = DataLoader(dataset, batch_size=16)

    #  Enable GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    start_time = time.time()
    correct, total = 0, 0
    
    with torch.no_grad():
        for batch in dataloader:
            #  Handle multiple input fields correctly
            if isinstance(text_key, tuple):  # RTE & PIQA
                inputs = tokenizer(*[batch[key] for key in text_key], return_tensors="pt", padding=True, truncation=True).to(device)
            else:
                inputs = tokenizer(batch[text_key], return_tensors="pt", padding=True, truncation=True).to(device)

            labels = torch.tensor(batch["label"]).to(device)

            outputs = model(**inputs)
            predictions = torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    
    end_time = time.time()
    accuracy = (correct / total) * 100 if total > 0 else 0.0
    latency = end_time - start_time

    return accuracy, latency

#  Enable GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_opt_classifier(model_name):
    """Load the specified OPT model and tokenizer."""
    model = OPTForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer
@app.route("/find_best_configuration", methods=["POST"])
def find_best_configuration():
    """Find the best model configuration by evaluating 8 different configurations."""
    data = request.json
    model_name = data.get("model_name")
    dataset_name = data.get("dataset_name")
    criterion = data.get("criterion")

    if not model_name or not dataset_name or not criterion:
        return jsonify({"error": "Missing required fields"}), 400

    model, tokenizer = load_opt_classifier(model_name)

    best_value = float("-inf") if criterion == "accuracy" else float("inf")
    best_configuration = None

    for i in range(8):  # 8 different configurations
        case_name = f"CHAI-{i:03b}"
        model_variant = model
        applied_methods = []

        model_variant = apply_pruning(model_variant,tokenizer,dataset_name)
        applied_methods.append("Pruning")

        if i & 2:  # Apply targeted fine-tuning
            model_variant = main_chai_target(model_variant, tokenizer, dataset_name)
            applied_methods.append("Targeted Fine-Tuning")

        if i & 1:  # Apply knowledge distillation
            classifier_model, tokenizer = load_opt_classifier()
            model = chai_knowledgde_distillation_enhancement(model_variant, classifier_model, tokenizer,dataset_name)
            applied_methods.append("Knowledge Distillation")

        # Evaluate the model
        if criterion == "accuracy":
            value, _ = evaluate_model(model_variant, tokenizer, dataset_name)
        elif criterion == "latency":
            _, value = evaluate_model(model_variant, tokenizer, dataset_name)
        else:  # Model size
            value = get_model_size(model_variant)

        if (criterion == "accuracy" and value > best_value) or (criterion != "accuracy" and value < best_value):
            best_value = value
            best_configuration = case_name

    return jsonify({
        "best_configuration": best_configuration,
        "criterion": criterion,
        "best_value": best_value
    })

@app.route("/choose_configuration", methods=["POST"])
def choose_configuration():
    """Apply user-selected configurations and return the results."""
    data = request.json
    model_name = data.get("model_name")
    dataset_name = data.get("dataset_name")
    configurations = data.get("configurations")

    if not model_name or not dataset_name or not configurations:
        return jsonify({"error": "Missing required fields"}), 400

    def evaluate_model(model, tokenizer, dataset_name):
        """Evaluates model accuracy on the given dataset."""

        #  Dataset mappings for correct loading
        dataset_mapping = {
            "sst2": ("glue", "sst2", "validation", "sentence"),
            "rte": ("glue", "rte", "validation", ("sentence1", "sentence2")),  # RTE has two sentences
            "piqa": ("piqa", None, "validation", ("goal", "sol1", "sol2")),  # PIQA has different format
        }

        if dataset_name not in dataset_mapping:
            return {"error": f"Unsupported dataset: {dataset_name}"}

        dataset_source, dataset_subset, split_name, text_key = dataset_mapping[dataset_name]

        #  Load dataset with cache handling
        try:
            dataset = load_dataset(dataset_source, dataset_subset, split=split_name, cache_dir="./cache") if dataset_subset \
                else load_dataset(dataset_source, split=split_name, cache_dir="./cache")
        except Exception as e:
            return {"error": f"Error loading dataset: {str(e)}"}

        dataloader = DataLoader(dataset, batch_size=16)

        #  Enable GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        #  Ensure correct precision (avoid float16 on CPU)
        if device.type == "cpu":
            model.to(torch.float32)

        start_time = time.time()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in dataloader:
                #  Handle multiple input fields correctly
                if isinstance(text_key, tuple):  # RTE & PIQA
                    inputs = tokenizer(*[batch[key] for key in text_key], return_tensors="pt", padding=True, truncation=True).to(device)
                else:
                    inputs = tokenizer(batch[text_key], return_tensors="pt", padding=True, truncation=True).to(device)

                #  FIX: Correct tensor creation and movement
                labels = torch.tensor(batch["label"], dtype=torch.long).to(device, non_blocking=True)

                outputs = model(**inputs)
                predictions = torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        end_time = time.time()
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        latency = end_time - start_time

        return [accuracy, latency]
    model, tokenizer = load_opt_classifier(model_name)
    size1 = get_model_size(model)
    [accuracy1,latency1]= evaluate_model(model, tokenizer, dataset_name)

    def load_model_for_kd(model_name):
        if not os.path.exists("temp_model.pth"):
            raise FileNotFoundError(" temp_model.pth not found. Ensure chai-quant saves it before chai-kd.")

        model = torch.load("temp_model.pth")  # Load full model, not just state_dict
        print(" Model successfully loaded from temp_model.pth")
        
        return model


    applied_methods = []
    model = apply_pruning(model,tokenizer,dataset_name)
    path = "/Users/sreebhargavibalija/Desktop/postmanchai/temp_model.pth"
    torch.save(model.state_dict(), path)
    if not os.path.exists(path):
        raise FileNotFoundError(f" Model file not found at {path}. Ensure `torch.save()` executed successfully.")

    print(" Model saved as temp_model.pth")

    applied_methods.append("Pruning")
    print("got back")
    size2 = get_model_size(model)
    print("sree")
    [accuracy2,latency2] = evaluate_model(model, tokenizer, dataset_name)
    def adjust_model_for_kd(model):
            """
            Adjusts the model's architecture to ensure compatibility for knowledge distillation.
            This may include aligning input dimensions, freezing specific layers, or modifying attention heads.
            """
            for param in model.parameters():
                param.requires_grad = True  # Ensure gradients are enabled for training
            print(" Model adjusted for Knowledge Distillation")
            return model
    from transformers import AutoModelForSequenceClassification
    from transformers import AutoModelForSequenceClassification

    def kd_load_model_for_kd(model_name, path):
        """ Loads the correct model architecture and applies saved weights. """
        print(f" Loading {model_name} with saved weights from {path}")

        #  Ensure the correct model is initialized before loading weights
        teacher_model = AutoModelForSequenceClassification.from_pretrained(model_name)

        #  Move to CPU and ensure float32 for compatibility
        device = torch.device("cpu")
        teacher_model.to(device)
        teacher_model = teacher_model.float()

        #  Load weights with strict=False to ignore missing keys
        if os.path.exists(path):
            try:
                state_dict = torch.load(path, map_location=device)
                teacher_model.load_state_dict(state_dict, strict=False)  # Allows missing/broken keys
                print(" Successfully loaded model weights.")
            except Exception as e:
                print(f" Error loading model weights: {e}")
        else:
            print(f" Model file {path} not found. Using default model weights.")

        return teacher_model


    # Path for saving/loading the model
    path = "/Users/sreebhargavibalija/Desktop/postmanchai/temp_model.pth"

    # Ensure configurations list is defined




    if "chai-quant" in configurations:
        model = chai_quant_enhancement(model, tokenizer, dataset_name)
        applied_methods.append("Quantization")
        print("applied_methods.append(Quantization)")

        # Ensure model dimensions match before saving
        # model = adjust_model_for_kd(model)  
        torch.save(model.state_dict(),path)
        print(" Model saved as temp_model.pth")

    if "chai-kd" in configurations:
        #  Check if model file exists before loading
        if os.path.exists(path):
            try:
                model.load_state_dict(torch.load(path))  #  Load model correctly
                print(" Loaded temp_model.pth before Knowledge Distillation")
            except Exception as e:
                print(f" Error loading model before KD: {e}")
        else:
            print(f" Model file {path} not found. Skipping initial load.")

        #  Load the classifier model
        classifier_model = kd_load_model_for_kd(model_name, path)

        #  Apply Knowledge Distillation Enhancement
        model = chai_knowledge_distillation_enhancement(model, classifier_model, tokenizer, dataset_name)

        #  Track applied methods
        applied_methods.append("Knowledge Distillation")
        print(f" Knowledge Distillation applied, Model shape: {get_model_size(model)}")

        #  Save the updated model state
        torch.save(model.state_dict(), path)
        print(f" Updated model saved at {path}")

    if "chai-target" in configurations:
        # Load the model after chai-quant modifications
        model.load_state_dict(torch.load(path))  #  Load model correctly
        print(" Loaded temp_model.pth before Targeted Fine-Tuning")
        model = main_chai_target(model, tokenizer, dataset_name)
        applied_methods.append("Targeted Fine-Tuning")
        print(f" Targeted Fine-Tuning applied, Model shape: {get_model_size(model)}")
        applied_methods.append("Targeted Fine-Tuning")
        torch.save(model.state_dict(),path)

    size3 = get_model_size(model)

    def evaluate_model(model, tokenizer, dataset_name):
        """Evaluates model accuracy on the given dataset."""

        #  Dataset mappings for correct loading
        dataset_mapping = {
            "sst2": ("glue", "sst2", "validation", "sentence"),
            "rte": ("glue", "rte", "validation", ("sentence1", "sentence2")),  # RTE has two sentences
            "piqa": ("piqa", None, "validation", ("goal", "sol1", "sol2")),  # PIQA has different format
        }

        if dataset_name not in dataset_mapping:
            return {"error": f"Unsupported dataset: {dataset_name}"}

        dataset_source, dataset_subset, split_name, text_key = dataset_mapping[dataset_name]

        #  Load dataset with cache handling
        try:
            dataset = load_dataset(dataset_source, dataset_subset, split=split_name, cache_dir="./cache") if dataset_subset \
                else load_dataset(dataset_source, split=split_name, cache_dir="./cache")
        except Exception as e:
            return {"error": f"Error loading dataset: {str(e)}"}

        dataloader = DataLoader(dataset, batch_size=16)

        #  Enable GPU if available
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        #  Ensure correct precision (avoid float16 on CPU)
        if device.type == "cpu":
            model.to(torch.float32)

        start_time = time.time()
        correct, total = 0, 0

        with torch.no_grad():
            for batch in dataloader:
                #  Handle multiple input fields correctly
                if isinstance(text_key, tuple):  # RTE & PIQA
                    inputs = tokenizer(*[batch[key] for key in text_key], return_tensors="pt", padding=True, truncation=True).to(device)
                else:
                    inputs = tokenizer(batch[text_key], return_tensors="pt", padding=True, truncation=True).to(device)

                #  FIX: Correct tensor creation and movement
                labels = torch.tensor(batch["label"], dtype=torch.long).to(device, non_blocking=True)

                outputs = model(**inputs)
                predictions = torch.argmax(F.softmax(outputs.logits, dim=-1), dim=-1)

                correct += (predictions == labels).sum().item()
                total += labels.size(0)

        end_time = time.time()
        accuracy = (correct / total) * 100 if total > 0 else 0.0
        latency = end_time - start_time

        return [accuracy, latency]


    [accuracy3,latency3]= evaluate_model(model, tokenizer, dataset_name)
    return jsonify({
        "configuration": applied_methods,
        "initial_accuracy": accuracy1,
        "base_accuracy(chai)": accuracy2,
        "Accuracy after adding these"+ str(applied_methods): accuracy3,
        "Initial_latency": latency1,
        "chai latency":latency2,
        "Latency after adding these"+ str(applied_methods): latency3,     
        "size decrement percnetage with intial model":(size2-size1)*100/size1,
        "size decrenment comparing with chai": (size3-size2)*100/size2,   
    })

@app.route("/")
def home():
    """Render the HTML page."""
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True,port= 5009)
