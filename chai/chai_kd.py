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

#  Detect and use the available device (MPS for Apple GPUs, CUDA for NVIDIA, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f" Using device: {device}")

def load_teacher_model(model_name, model_path):
    """ Load the teacher model and move it to the correct device """
    print(f" Loading teacher model {model_name} from {model_path}")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f" Model file not found at {model_path}")

    teacher_model = OPTForSequenceClassification.from_pretrained(model_name)

    try:
        teacher_model.load_state_dict(torch.load(model_path, map_location=device))
        print(" Successfully loaded teacher model weights.")
    except Exception as e:
        print(f" Error loading teacher model weights: {e}")

    teacher_model.to(device)
    teacher_model.eval()
    return teacher_model

def chai_knowledge_distillation_enhancement(student_model, teacher_model, tokenizer, dataset_name):
    """ Apply Knowledge Distillation (CHAI-KD) Enhancement """
    
    print("\n Applying Knowledge Distillation (CHAI-KD)...")
    print(" [chai-kd] Before modification: model parameters")
    for name, param in student_model.named_parameters():
        print(f"  {name}: {param.shape}")

    #  Move student model to the correct device
    student_model.to(device)

    #  Training hyperparameters
    epochs = 1
    batch_size = 16
    temperature = 2.0
    alpha = 0.5

    #  Load dataset and tokenize it
    dataset = load_dataset("glue", "sst2", split="train[:5000]")
    tokenized_dataset = dataset.map(
        lambda e: tokenizer(e['sentence'], truncation=True, padding='max_length', max_length=128),
        batched=True
    )

    training_args = TrainingArguments(
        output_dir="./chai_kd_model",
        learning_rate=5e-5,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        save_strategy="epoch",
        report_to="none"
    )

    class KDTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            """ Compute Knowledge Distillation Loss """

            labels = inputs.pop("labels")  # Extract labels
            inputs = {k: v.to(device) for k, v in inputs.items()}  #  Move inputs to the correct device
            model.to(device)

            student_outputs = model(**inputs)
            student_logits = student_outputs.logits

            with torch.no_grad():
                teacher_outputs = teacher_model(**inputs)  #  Ensure teacher model is on the correct device
                teacher_logits = teacher_outputs.logits

            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction="batchmean"
            ) * (temperature ** 2)

            ce_loss = F.cross_entropy(student_logits, labels)
            loss = alpha * kd_loss + (1 - alpha) * ce_loss  # Combined loss

            return (loss, student_outputs) if return_outputs else loss

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding="longest")

    trainer = KDTrainer(
        model=student_model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator  #  Ensure proper padding
    )

    trainer.train()

    print(" [chai-kd] After modification: model parameters")
    for name, param in student_model.named_parameters():
        print(f"  {name}: {param.shape}")

    return student_model
