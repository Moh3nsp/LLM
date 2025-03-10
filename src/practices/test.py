import re
from tokenizer.tokenizer_v2 import TokenizerV2

all_words= 'Hi this is a test full of words and jokes <|unk|>'
preprocessed = re.split(r'([,.:;?_!"()\']|--|\s)', all_words)
preprocessed = [item.strip() for item in preprocessed if item.strip()]
vocab = {token:integer for integer,token in enumerate(preprocessed)}

tokenizer = TokenizerV2(vocab)
ids =tokenizer.encode('Hi this')

print(ids)
print(tokenizer.decode(ids))