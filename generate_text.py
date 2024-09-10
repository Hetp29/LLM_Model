from transformers import GPT2LMHeadModel, GPT2Tokenizer  # type: ignore

# Load the fine-tuned model and tokenizer
model = GPT2LMHeadModel.from_pretrained("./fine_tuned_model")
tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_model")

# Generate text
input_text = "Once upon a time"
inputs = tokenizer.encode(input_text, return_tensors="pt")

# Use top-k and top-p sampling to introduce diversity
outputs = model.generate(
    inputs, 
    max_length=100, 
    num_return_sequences=1, 
    do_sample=True,         # Enable sampling
    top_k=50,               # Consider the top 50 words by probability at each step
    top_p=0.95,             # Use nucleus sampling (top-p) for more variety
    temperature=0.7         # Control randomness: lower is less random, higher is more random
)

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
