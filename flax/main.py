import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
from flax import nnx
from trainer import TranslationTrainer
from data.dataloader import Dataloader
from instance import TransformerConfig, build_transformer

def get_batches(src, tgt, batch_size, key):
    indices = jax.random.permutation(key, len(src))

    full_batch_count = len(src) // batch_size

    for batch_index in range(full_batch_count):
        start = batch_index * batch_size
        end = start + batch_size

        batch_ids = indices[start:end]

        src_batch = src[batch_ids]
        tgt_batch = tgt[batch_ids]

        yield src_batch, tgt_batch

def vector2sentence(index2word, vectors):
    sentences = []
    for vector in vectors:
        words = []
        for index in vector:
            word = index2word[int(index)]
            if word == 'EOS':
                break
            if word not in {'SOS','PAD'}:
                words.append(word)

        sentences.append(' '.join(words))
    return sentences

if __name__ == '__main__':
    epochs = 20

    dataloader = Dataloader(dir='flax/data')
    dataloader.preparing_data()
    word2index, index2word = dataloader.get_dict()
    vocab_size = len(word2index)
    src, tgt = dataloader.get_src(), dataloader.get_tgt()
    dtrain, dval, dtest = dataloader.split_data(key = jax.random.PRNGKey(0))

    key = jax.random.PRNGKey(1)

    config = TransformerConfig()
    model = build_transformer(src_vocab_size=vocab_size, tgt_vocab_size=vocab_size, 
                              max_len=dataloader.max_len, config=config, rngs=nnx.Rngs(params=0, dropout=1))
    
    trainer = TranslationTrainer(model=model, 
                                 embedding_size=config.embedding_size, 
                                 warmup_steps=100)
    for i in range(epochs):

        key, train_key, val_key = jax.random.split(key, 3)
        
        loss_train = 0
        n_batches = 0
        for src_train, tgt_train in get_batches(
            src=dtrain[0], tgt=dtrain[1],
            batch_size=16, key=train_key,
        ):
            loss_train += trainer.training_step(src_train, tgt_train)
            n_batches+=1
        loss_train = loss_train / n_batches


        loss_val = 0
        n_batches = 0
        for src_val, tgt_val in get_batches(
                    src=dval[0], tgt=dval[1],
                    batch_size=16, key=val_key,
        ):
            loss_val += trainer.validation_step(src_val, tgt_val)
            n_batches +=1
        loss_val =loss_val/n_batches
        pred= trainer.do_prediction(src_val[:1],seq_length=dataloader.max_len)
        
        print('Epoch:', i)
        print('training loss:', loss_train)
        print('validation loss:', loss_val)

        print('source:    ', vector2sentence(index2word, src_val[:1]))
        print('prediction:', vector2sentence(index2word,pred))
        print('target:    ', vector2sentence(index2word, tgt_val[:1]))

    _, state = nnx.split(model)
    #nnx.display(state)

    checkpointer = ocp.StandardCheckpointer()
    checkpoint_path = '/Users/ryxx/24SS-FlaxEvo/flax/checkpoints/state'
    checkpointer.save(checkpoint_path, state)
    checkpointer.wait_until_finished()

        
        
        

       




    
