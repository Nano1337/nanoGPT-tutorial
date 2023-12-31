
if __name__ == "__main__": 
    with open('input.txt', 'r', encoding='utf-8') as f: 
        text = f.read()

        print(f'length of dataset in chars {len(text)}')

        # counting and showing unique characters found in text 
        chars = sorted(list(set(text)))
        print(f'Number of unique characters: {len(chars)}')
        vocab = ''.join(chars)
        print(f'Unique characters are: {vocab}')

        # create tokenizer
        stoi = {ch:i for i, ch in enumerate(chars)}
        itos = {i:ch for i, ch in enumerate(chars)}
        encode = lambda s: [stoi[c] for c in s] # s is the input string
        decode = lambda l: [itos[i] for i in l] # l is the encoded/tokenized/serialized string

        print(encode("hii there"))
        print(decode(encode("hii there")))