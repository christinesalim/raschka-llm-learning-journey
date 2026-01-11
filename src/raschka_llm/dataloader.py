from pathlib import Path
from typing import List
import torch
import tiktoken

from torch.utils.data import Dataset, DataLoader



#print("Pytorch version: ", torch.__version__)

class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        
        #Tokenize the entire text
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})
        assert len(token_ids) > max_length, "Number of tokenized inputs must be at least equal to max_length+1"
        
        #Use a sliding window to chunk the book into overlapping sequences of max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i+1:i+max_length+1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
        #Initialize the tokenizer
        tokenizer = tiktoken.get_encoding("gpt2")
        
        #Create dataset
        dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
        
        #Create dataloader
        dataloader = DataLoader(
            dataset,
            batch_size = batch_size,
            shuffle = shuffle,
            drop_last = drop_last,
            num_workers = num_workers
        )
        return dataloader
    
    
with open("../../data/the-verdict.txt", "r", encoding="utf-8") as f:
        raw_text = f.read()
    
#stride of 1 moves the input field by 1 position
#batch_size = number of samples grouped for training
dataloader = create_dataloader_v1(
        raw_text, batch_size=1, max_length=4, stride=1, shuffle=False
)

data_iter = iter(dataloader)
first_batch = next(data_iter)
print(first_batch)

second_batch = next(data_iter)
print(second_batch)


dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs:\n", inputs)
print("\nTargets:\n", targets)

dataloader = create_dataloader_v1(raw_text, batch_size=6, max_length=2, stride=2, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs: \n", inputs)
print("\nTargets:\n", targets)

dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
data_iter = iter(dataloader)
inputs, targets = next(data_iter)

print("Inputs: \n", inputs)
print("\nTargets:\n", targets)



vocab_size = 50257
output_dim = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(raw_text, batch_size=8, max_length=max_length, 
                                  stride = max_length, shuffle=False)

data_iter = iter(dataloader)
inputs, targets = next(data_iter)
print("\nInputs shape:", inputs.shape) #[8,4] 8 samples and 4 tokens in each row
token_embeddings = token_embedding_layer (inputs)
print(token_embeddings.shape)

context_length = max_length

context_length = max_length

#Converts integer indices into dense vectors
#context_length = 4 number of positions
#output_dim = size of each embedding vector

#Creates a learnable lookup table with 4 positions, each mapped to a 256-dim vector
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)

#torch.arange(4) returns a tensor [0,1,2,3]
pos_embeddings = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
