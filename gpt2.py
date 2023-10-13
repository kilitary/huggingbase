from transformers import AutoModelForCausalLM, AutoTokenizer

device = "cpu"  # the device to load the model onto

model = AutoModelForCausalLM.from_pretrained("https://huggingface.co/microsoft/phi-1_5/raw/main/config.json")
tokenizer = AutoTokenizer.from_pretrained("https://huggingface.co/microsoft/phi-1_5/raw/main/config.json")

text = "<sWhat is your favourite condiment?"
"Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!</s> "
"Do you have mayonnaise recipes?"

encodeds = tokenizer(text, return_tensors="pt", add_special_tokens=False)

model_inputs = encodeds.to(device)
model.to(device)

generated_ids = model.generate(**model_inputs, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])
