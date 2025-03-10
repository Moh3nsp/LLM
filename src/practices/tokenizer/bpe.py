#https://github.com/openai/tiktoken

import tiktoken

tokenizer = tiktoken.get_encoding('gpt2')

text = ("Hello, do you like tea? <|endoftext|> In the sunlit terraces of some unknown Place.")

ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
print(ids)
print(tokenizer.)