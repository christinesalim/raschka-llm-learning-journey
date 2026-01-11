"""BytePair Encoding tokenizer using tiktoken 

BPE tokenizer allows the model to break down words that aren't predefined in the vocabulary into
smaller subword units or even individual characters, allowing it to handle out-of-vocabulary
words

This BPE tokenizer is from OpenAI's open-source tiktoken library
"""

import importlib.metadata
import tiktoken

print("toktoken version:", importlib.metadata.version("tiktoken"))

tokenizer = tiktoken.get_encoding("gpt2")

text = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    "of someunknownPlace."
)


#Run the encoder
integers = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
print (integers)


#Run the decoder
strings = tokenizer.decode(integers)
print(strings)


#Encode the story from the-verdict.txt
with open ("../../data/the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
    
enc_text = tokenizer.encode(raw_text)
print(f"enc_text len = ", len(enc_text))

enc_sample = enc_text[50:]

context_size = 4

x = enc_sample[:context_size]
y = enc_sample[1:context_size+1] #slide forward by 1 token

print(f"x: {x}")
print(f"y:      {y}")

#Prediction would like as follows:
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i] #next word
    
    print(context, "---->", desired)
    
    
#Decode to strings
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    
    print(tokenizer.decode(context), "---->", tokenizer.decode([desired]))