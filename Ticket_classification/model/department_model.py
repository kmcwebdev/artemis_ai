from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

def load_model():
    model_path = 'microsoft/deberta-v3-small'
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
    # department_model = AutoModelForSequenceClassification.from_pretrained(model_path)

load_model()
