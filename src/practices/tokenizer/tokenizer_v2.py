import re

class TokenizerV2:
    def __init__(self, vocab_dic):
        self.str_to_int= vocab_dic
        self.int_to_str = {index:vocab for vocab, index in vocab_dic.items()}
            
    def encode(self, text):
        preprocessed= re.split(r'([,.:;?_!"()\']|--|\s)', text)
        preprocessed= [item.strip() for item in preprocessed if item.strip()]
        #Replaces (unknown words by <|unk|> tokens)
        preprocessed = [item if item in self.str_to_int else '<|unk|>' for item in preprocessed]
        ids= [self.str_to_int[vocab] for vocab in preprocessed]
        return ids
        
    def decode(self, ids):
        text= ' '.join([self.int_to_str[id] for id in ids])
         #Replaces spaces before the specified punctuations
        text=  re.sub(r'\s+([,.:;?!"()\'])', r'\1', text)
        return text