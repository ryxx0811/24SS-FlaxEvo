import jax
import jax.numpy as jnp
import os

PAD = 0
sOS = 1
EOS = 2
max_len = 10

class Dataloader():
    def __init__(self, dir: str, max_len: int = max_len):
        self.dir = dir
        self.max_len = int(max_len)
        self._word2index = {'PAD':0,
                           'SOS':1, 
                           'EOS':2}
        self._index2word = {0:'PAD', 
                           1:'SOS', 
                           2:'EOS'}
        self._src_sentences = None
        self._tgt_sentences = None
        self._src = None
        self._tgt = None

    def _get_sentences(self):
        
        for filename in os.listdir(self.dir):
            if filename.endswith('.en'):
                file_path = os.path.join(self.dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    src_sentences = file.read().splitlines()
            elif filename.endswith('.de'):
                file_path = os.path.join(self.dir, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    tgt_sentences = file.read().splitlines()

        self._src_sentences = [f'SOS {src} EOS' for src in src_sentences]
        self._tgt_sentences = [f'SOS {tgt} EOS' for tgt in tgt_sentences]
    
        #return src_sentences,tgt_sentences

    def preparing_data(self):
        self._get_sentences()
        sentences = self._src_sentences + self._tgt_sentences

        for sentence in sentences:
            for word in sentence.split():
                if word not in self._word2index:
                    token_id = len(self._word2index)
                    self._word2index[word] = token_id
                    self._index2word[token_id] = word

    def sentence2vector(self, sentences):
        vectors = []
        for sentence in sentences:
            words = sentence.split()
            vector = [self._word2index[word] for word in words]
            if len(vector) > self.max_len:
                raise ValueError(f'sentence has {len(vector)} tokens; max_len is {self.max_len}')
            padding_length = self.max_len - len(vector)
            vector += [self._word2index['PAD']] * padding_length
            vectors.append(vector)
                
        return jnp.array(vectors, dtype=jnp.int32)
    

    def get_src(self):
        self._src = self.sentence2vector(self._src_sentences)
        return self._src

    def get_tgt(self):
            self._tgt = self.sentence2vector(self._tgt_sentences)
            return self._tgt
    def get_dict(self):
        return self._word2index,self._index2word

    def split_data(self, key, train_ratio=0.6, val_ratio=0.2):
        '''Split paired source/target arrays into train, validation, and test sets.'''
        if len(self._src) != len(self._tgt):
            raise ValueError('src and tgt must contain the same number of sentences')
        
        indices = jax.random.permutation(key, len(self._src))
        
        train_end = int(len(self._src) * train_ratio)
        val_end = train_end + int(len(self._src) * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        return (
            (self._src[train_indices], self._tgt[train_indices]),
            (self._src[val_indices], self._tgt[val_indices]),
            (self._src[test_indices], self._tgt[test_indices]),
        )
    





    
        

    




        

    

  
