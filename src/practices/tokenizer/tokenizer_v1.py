import re

class TokenizerV1:
    def __init__(self, vocab_dic):
        self.str_to_int= vocab_dic
        self.int_to_str = {index:vocab for vocab, index in vocab_dic}
        
    def encode(self, text):
        """convert text to ids"""
        #tokenizing text
        preprocessed = re.split(r'([,.?_!"()\']|--|\s)', text)
        preprocessed= [item.strip() for item in preprocessed if item.strip()]
        ids= [self.str_to_int[item] for item in preprocessed]        
        return ids
        
    
    def decode(self, ids):
        text = ' '.join([self.int_to_str[id] for id in ids])
        text= re.sub(r'\s+([,.?!"()\'])', r'\1',text)   
        return text
             