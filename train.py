from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments # type: ignore
from datasets import load_dataset # type: ignore

model_name = "gpt2"  # Pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Set the pad token to be the eos token
tokenizer.pad_token = tokenizer.eos_token

data_path = "data/my_data.txt"

# Tokenize the dataset and add labels
def load_data(file_path):
    dataset = load_dataset('text', data_files={'train': file_path})
    
    def tokenize_function(examples):
        # Tokenize the input text and use the input IDs as labels
        tokenized_inputs = tokenizer(examples["text"], truncation=True, padding="max_length", max_length=128)
        tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  # Labels are the same as input_ids
        return tokenized_inputs
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    return tokenized_dataset

def fine_tune_model():
    tokenized_dataset = load_data(data_path)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./output",
        overwrite_output_dir=True,
        per_device_train_batch_size=2,
        num_train_epochs=3,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        remove_unused_columns=False  # Ensure columns aren't removed
    )
    
    # Trainer setup
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset['train']
    )
    
    trainer.train()  # Train the model
    
    model.save_pretrained("./fine_tuned_model")  # Save the fine-tuned model
    tokenizer.save_pretrained("./fine_tuned_model")

if __name__ == "__main__":
    fine_tune_model()
