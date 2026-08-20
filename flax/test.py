import orbax.checkpoint as ocp
from flax import nnx

from data.dataloader import Dataloader
from instance import TransformerConfig, build_transformer
from trainer import TranslationTrainer


dataloader = Dataloader(dir='flax/data')
dataloader.preparing_data()

word2index, index2word = dataloader.get_dict()
vocab_size = len(word2index)

# Must match the architecture used when saving.
config = TransformerConfig()

model = build_transformer(
    src_vocab_size=vocab_size,
    tgt_vocab_size=vocab_size,
    max_len=dataloader.max_len,
    config=config,
    rngs=nnx.Rngs(params=999, dropout=998),
)

# Get the Transformer structure and an empty state template.
graphdef, empty_state = nnx.split(model)

# Replace random weights with saved weights.
checkpointer = ocp.StandardCheckpointer()

saved_state = checkpointer.restore(
    '/Users/ryxx/24SS-FlaxEvo/flax/checkpoints/state',
    target=empty_state,
)

model = nnx.merge(graphdef, saved_state)


