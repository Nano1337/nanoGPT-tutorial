import torch 

def tokenize(path): 
    
    with open(path, 'r', encoding='utf-8') as f: 
        text = f.read()

        chars = sorted(list(set(text)))

        stoi = {ch:i for i, ch in enumerate(chars)}

        encode = lambda s: [stoi[c] for c in s]

        data = torch.tensor(encode(text))

        n = int(0.9*len(data))
        train_data, val_data = data[:n], data[n:]

        return train_data, val_data



    
if __name__ == "__main__": 
    train_data, val_data = tokenize('input.txt')
    print(val_data.shape)
