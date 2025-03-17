# Fine-Tuning LLMs with LoRA and Adapters

## Description
This project demonstrates fine-tuning of Large Language Models (LLMs) using two techniques:
1. **LoRA (Low-Rank Adaptation)**: A parameter-efficient fine-tuning method that freezes pre-trained weights and trains only low-rank updates.
2. **Adapters**: A modular fine-tuning approach where small trainable adapter layers are added to the pre-trained model.

Additionally, this implementation logs loss values for tracking training performance.

## Installation
To run this project, install the required libraries:
```bash
pip install transformers datasets peft adapter-transformers
```

## Code Implementation

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from peft import LoraConfig, get_peft_model, TaskType
from adapter_transformers import AutoAdapterModel, PfeifferConfig
from datasets import load_dataset
```
  
Ensure you have the required dependencies installed:  
```bash
pip install transformers datasets matplotlib peft

from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback
from datasets import load_dataset
import matplotlib.pyplot as plt

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:1%]")

# Load model & tokenizer
model_name = "EleutherAI/pythia-70m"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Tokenization
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()
    return tokenized_inputs

dataset = dataset.map(tokenize_function, batched=True)

# Training Arguments
training_args = TrainingArguments(
    output_dir="./results_finetuning",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    save_strategy="epoch",
    logging_steps=10,
    report_to="none"
)

# Logging loss
loss_values = []
class LossLoggerCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "loss" in logs:
            loss_values.append(logs["loss"])
            print(f"Step {state.global_step}: Loss = {logs['loss']}")

# Trainer (Full Fine-Tuning)
trainer = Trainer(model=model, args=training_args, train_dataset=dataset, callbacks=[LossLoggerCallback()])
trainer.train()

# Plot Loss Curve
plt.plot(loss_values, marker="o", label="Full Fine-Tuning")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.legend()
plt.show()

![LORa](https://github.com/user-attachments/assets/a415736c-02ce-4402-be94-d8349251ad1a)



## Summary
This project demonstrates two approaches for fine-tuning LLMs:
- **LoRA**: Efficient low-rank adaptation
- **Adapters**: Modular and reusable fine-tuning

Each method logs loss values to track performance.

## Future Work
- Implement quantization for reducing memory footprint
- Experiment with different fine-tuning strategies for optimization
