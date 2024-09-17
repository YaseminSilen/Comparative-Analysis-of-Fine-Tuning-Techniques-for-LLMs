import torch
from transformers import GPT2ForSequenceClassification, GPT2Tokenizer, TrainingArguments, Trainer, TrainerCallback
from datasets import Dataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from transformers import get_scheduler
import seaborn as sns
import time
import psutil
import json

# 1. 加载预训练模型和分词器
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2ForSequenceClassification.from_pretrained(model_name, num_labels=3)
model.config.pad_token_id = model.config.eos_token_id

# 2. 准备数据集
def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)
    dataset = Dataset.from_pandas(df)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_dataset = tokenized_dataset.rename_column("label", "labels")
    
    return tokenized_dataset

# 加载训练集和验证集
train_dataset = load_and_preprocess_data("./tweet_financial_training_set_final.csv")
val_dataset = load_and_preprocess_data("./tweet_financial_validation_set_final.csv")

print("Train dataset columns:", train_dataset.column_names)
print("Validation dataset columns:", val_dataset.column_names)

# 3. 定义评估指标
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.Tensor(logits), dim=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
    acc = accuracy_score(labels, predictions)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
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

# 设置训练参数
training_args = TrainingArguments(
    output_dir="./gpt2_finance_sentiment_baseline",
    overwrite_output_dir=True,
    num_train_epochs=5,  # 增加到 3 个 epoch
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    evaluation_strategy="steps",  # 使用 "steps" 策略
    eval_steps=100,  # 每 100 步评估一次
    save_strategy="steps",
    save_steps=100,  # 每 100 步保存一次
    save_total_limit=2,  # 只保存最后两个检查点
    load_best_model_at_end=True,
    learning_rate=2e-4,  # 降低学习率
    weight_decay=0.01,
    metric_for_best_model="accuracy",
    logging_dir='./logs',
    logging_steps=100,
    warmup_steps=300,  # 添加预热步骤
    fp16=True,  # 如果您的 GPU 支持，启用 16 位精度
    gradient_accumulation_steps=2,  # 梯度累积，实际batch_size = 8 * 4 = 32
)


#添加资源监控    
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
        # CPU 使用率
        self.cpu_usage.append(psutil.cpu_percent())

        # 内存使用率
        memory = psutil.virtual_memory()
        self.memory_usage.append(memory.percent)

        # GPU 内存使用 (如果可用)
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            self.gpu_memory_usage.append(gpu_memory * 100)  # 转换为百分比

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


# 5. 创建Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[metrics_callback, resource_monitor],
)

# 6. 开始训练
trainer.train()

# 7. 保存模型
trainer.save_model("./gpt2_finance_sentiment_baseline")

# 获取资源使用摘要
resource_summary = resource_monitor.get_summary()

# 8. 使用模型进行预测
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    sentiment_map = {0: "Negative", 1: "Neutral", 2: "Positive"}
    return sentiment_map[predicted_class]



# 绘制学习率曲线
plt.figure(figsize=(10, 5))
plt.plot(metrics_callback.lr_schedule)
plt.title('Learning Rate Schedule')
plt.xlabel('Training Steps')
plt.ylabel('Learning Rate')
plt.savefig('learning_rate_curve.png')
plt.close()

# 绘制训练损失和验证损失曲线
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

# 绘制评估指标曲线
metrics = ['accuracy', 'f1', 'precision', 'recall']
plt.figure(figsize=(12, 8))
for metric in metrics:
    values = [log[f'eval_{metric}'] for log in metrics_callback.eval_metrics]
    plt.plot(values, label=metric.capitalize())
plt.title('Evaluation Metrics')
plt.xlabel('Epochs')
plt.ylabel('Score')
plt.legend()
plt.savefig('evaluation_metrics.png')
plt.close()

# [预测部分的代码保持不变]

# 添加混淆矩阵


def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

# 在验证集上进行预测
val_predictions = trainer.predict(val_dataset)
y_true = val_predictions.label_ids
y_pred = np.argmax(val_predictions.predictions, axis=1)

# 绘制混淆矩阵
plot_confusion_matrix(y_true, y_pred)

# 打印最终的评估指标
final_metrics = compute_metrics((val_predictions.predictions, y_true))
print("Final Evaluation Metrics:")
for metric, value in final_metrics.items():
    print(f"{metric.capitalize()}: {value:.4f}")

# 测试模型
test_texts = [
    "The company reported strong earnings, exceeding market expectations.",
    "The stock price plummeted after the disappointing quarterly results.",
    "The market remained stable despite global economic uncertainties."
]

for text in test_texts:
    print(f"Text: {text}")
    print(f"Predicted sentiment: {predict_sentiment(text)}\n")
    
# 打印摘要
print("Resource Usage Summary:")
print(f"Total training time: {resource_summary['total_time']:.2f} seconds")
print(f"Average CPU Usage: {resource_summary['avg_cpu_usage']:.2f}%")
print(f"Max CPU Usage: {resource_summary['max_cpu_usage']:.2f}%")
print(f"Average Memory Usage: {resource_summary['avg_memory_usage']:.2f}%")
print(f"Max Memory Usage: {resource_summary['max_memory_usage']:.2f}%")
if resource_summary['avg_gpu_memory_usage'] is not None:
    print(f"Average GPU Memory Usage: {resource_summary['avg_gpu_memory_usage']:.2f}%")
    print(f"Max GPU Memory Usage: {resource_summary['max_gpu_memory_usage']:.2f}%")

# 保存资源使用数据到文件

with open('resource_usage.json', 'w') as f:
    json.dump(resource_summary, f)

# 可视化资源使用情况

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
