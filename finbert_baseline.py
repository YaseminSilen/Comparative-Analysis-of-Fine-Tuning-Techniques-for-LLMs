import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch import nn
import seaborn as sns
import time
import psutil
import json

# 1. load model and tokenizer
model_name = "ProsusAI/finbert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

# 2. load and preprocess data
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    return tokenized_dataset

# Load  training and validation datasets
train_dataset = load_and_preprocess_data("./FPB_training_set_final.csv")
val_dataset = load_and_preprocess_data("./FPB_validation_set_final.csv")

print("Train dataset columns:", train_dataset.column_names)
print("Validation dataset columns:", val_dataset.column_names)

# 3. Define class weights
class_weights = torch.tensor([1.2, 1.2, 1.0], dtype=torch.float)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_weights = class_weights.to(device)

# 4. Define evaluation metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }

class MetricsCallback(TrainerCallback):
    def __init__(self):
        self.train_losses = []
        self.eval_losses = []
        self.eval_metrics = []
        self.lr_schedule = []

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            if 'loss' in logs:
                self.train_losses.append(logs['loss'])
            if 'eval_loss' in logs:
                self.eval_losses.append(logs['eval_loss'])
            if 'eval_accuracy' in logs:
                self.eval_metrics.append(logs)
            if 'learning_rate' in logs:
                self.lr_schedule.append(logs['learning_rate'])

metrics_callback = MetricsCallback()

# 5.training_args
training_args = TrainingArguments(
    output_dir="./finbert_finance_sentiment_baseline",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
    learning_rate=2e-5,  # 通常全参数微调使用较小的学习率
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    logging_dir='./logs',
    logging_steps=100,
    warmup_steps=100,
    fp16=True,
    gradient_accumulation_steps=2,
)

# loss function
class WeightedLossTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


#add resource monitor  
class ResourceMonitor(TrainerCallback):
    def __init__(self):
        self.cpu_usage = []
        self.memory_usage = []
        self.gpu_memory_usage = []
        self.start_time = None
        self.end_time = None

    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()

    def on_train_end(self, args, state, control, **kwargs):
        self.end_time = time.time()

    def on_step_end(self, args, state, control, **kwargs):
        # CPU 
        self.cpu_usage.append(psutil.cpu_percent())

      # memory usage
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.percent)

          # GPU memory usage
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.gpu_memory_usage.append(gpu_memory * 100)  

    def get_summary(self):
        total_time = self.end_time - self.start_time
        return {
            "total_time": total_time,
            "avg_cpu_usage": sum(self.cpu_usage) / len(self.cpu_usage),
            "max_cpu_usage": max(self.cpu_usage),
            "avg_memory_usage": sum(self.memory_usage) / len(self.memory_usage),
            "max_memory_usage": max(self.memory_usage),
            "avg_gpu_memory_usage": sum(self.gpu_memory_usage) / len(self.gpu_memory_usage) if self.gpu_memory_usage else None,
            "max_gpu_memory_usage": max(self.gpu_memory_usage) if self.gpu_memory_usage else None,
        }


resource_monitor = ResourceMonitor()

#create trainer
trainer = WeightedLossTrainer(
    class_weights=class_weights,
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback, resource_monitor],
)

# train
trainer.train()

# save model
trainer.save_model("./finbert_finance_sentiment_baseline")


# obtain resource usage summary
resource_summary = resource_monitor.get_summary()

# 7. Evaluation
device = "cuda" if torch.cuda.is_available() else "cpu"
model = AutoModelForSequenceClassification.from_pretrained("./finbert_finance_sentiment_baseline").to(device)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[predicted_class]

# learning rate schedule
plt.figure(figsize=(10, 5))
plt.plot(metrics_callback.lr_schedule)
plt.title('Learning Rate Schedule')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.savefig('learning_rate_curve.png')
plt.close()

# loss curves
plt.figure(figsize=(10, 5))
plt.plot(metrics_callback.train_losses, label='Training Loss')
plt.plot(np.linspace(0, len(metrics_callback.train_losses), len(metrics_callback.eval_losses)), 
         metrics_callback.eval_losses, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Training Steps')
plt.ylabel('Loss')
plt.legend()
plt.savefig('loss_curves.png')
plt.close()

# evaluation metrics
metrics = ['accuracy', 'f1', 'precision', 'recall']
plt.figure(figsize=(12, 8))
for metric in metrics:
    values = [log[f'eval_{metric}'] for log in metrics_callback.eval_metrics]
    plt.plot(values, label=metric.capitalize())
plt.title('Evaluation Metrics')
plt.xlabel('Evaluation Steps')
plt.ylabel('Score')
plt.legend()
plt.savefig('evaluation_metrics.png')
plt.close()

# confusion matrix
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()


val_predictions = trainer.predict(val_dataset)
y_true = val_predictions.label_ids
y_pred = np.argmax(val_predictions.predictions, axis=1)


plot_confusion_matrix(y_true, y_pred)

# final evaluation metrics
final_metrics = compute_metrics((val_predictions.predictions, y_true))
print("Final Evaluation Metrics:")
for metric, value in final_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# test predictions
test_texts = [
    "The company reported strong earnings, exceeding market expectations.",
    "The stock price plummeted after the disappointing quarterly results.",
    "The market remained stable despite global economic uncertainties."
]

for text in test_texts:
    print(f"Text: {text}")
    print(f"Predicted sentiment: {predict_sentiment(text)}\n")
    
    
# print resource usage summary
print("Resource Usage Summary:")
print(f"Total training time: {resource_summary['total_time']:.2f} seconds")
print(f"Average CPU Usage: {resource_summary['avg_cpu_usage']:.2f}%")
print(f"Max CPU Usage: {resource_summary['max_cpu_usage']:.2f}%")
print(f"Average Memory Usage: {resource_summary['avg_memory_usage']:.2f}%")
print(f"Max Memory Usage: {resource_summary['max_memory_usage']:.2f}%")
if resource_summary['avg_gpu_memory_usage'] is not None:
    print(f"Average GPU Memory Usage: {resource_summary['avg_gpu_memory_usage']:.2f}%")
    print(f"Max GPU Memory Usage: {resource_summary['max_gpu_memory_usage']:.2f}%")

# save resource usage summary to a JSON file

with open('resource_usage.json', 'w') as f:
    json.dump(resource_summary, f)


# plot resource usage

plt.figure(figsize=(12, 8))
plt.plot(resource_monitor.cpu_usage, label='CPU Usage')
plt.plot(resource_monitor.memory_usage, label='Memory Usage')
if resource_monitor.gpu_memory_usage:
    plt.plot(resource_monitor.gpu_memory_usage, label='GPU Memory Usage')
plt.title('Resource Usage During Training')
plt.xlabel('Training Steps')
plt.ylabel('Usage (%)')
plt.legend()
plt.savefig('resource_usage.png')
plt.close()