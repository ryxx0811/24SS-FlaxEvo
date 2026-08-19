'''Factory for constructing Transformer candidates.

Keep all architecture choices here.  `main.py` should only choose a
configuration, create a fresh random key, and call `build_transformer`.
'''

from dataclasses import dataclass

from flax import nnx

from transformer.Decoder import Decoder, DecoderLayer
from transformer.Dropout import Dropout
from transformer.Embedder import Embedder
from transformer.Encoder import Encoder, EncoderLayer
from transformer.FeedForward import FeedForward
from transformer.Linear import Linear
from transformer.MultiHeadAttention import MultiHeadAttention
from transformer.Norm import Norm
from transformer.PositionalEncoding import PositionalEncoding
from transformer.Transformer import Transformer


@dataclass(frozen=True)
class TransformerConfig:
    '''One architecture candidate for training or evolution.'''

    embedding_size: int = 64
    feed_forward_size: int = 256
    num_heads: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 2
    dropout_rate: float = 0.1

    def validate(self) -> None:
        if self.embedding_size % self.num_heads!= 0:
            raise ValueError('embedding_size must be divisible by num_heads')
        if self.num_encoder_layers < 1 or self.num_decoder_layers < 1:
            raise ValueError('the encoder and decoder must each have at least one layer')


def build_transformer(
    *,
    src_vocab_size: int,
    tgt_vocab_size: int,
    max_len: int,
    config: TransformerConfig,
    rngs: nnx.Rngs,
) -> Transformer:
    '''Create one fresh Transformer from one candidate configuration.

    Source and target vocabularies may currently be the same size because your
    data loader has one shared ``word2index`` dictionary.  Keeping them as two
    arguments lets you switch to separate vocabularies later without changing
    the model structure.
    '''
    config.validate()
    d_model = config.embedding_size

    def attention() -> MultiHeadAttention:
        return MultiHeadAttention(
            embedding_size=d_model,
            num_heads=config.num_heads,
            dropout=Dropout(p_dropout=config.dropout_rate, rngs=rngs),
            q=Linear(input_size=d_model, output_size=d_model, rngs=rngs),
            k=Linear(input_size=d_model, output_size=d_model, rngs=rngs),
            v=Linear(input_size=d_model, output_size=d_model, rngs=rngs),
            out=Linear(input_size=d_model, output_size=d_model, rngs=rngs),
        )

    def encoder_layer() -> EncoderLayer:
        return EncoderLayer(
            self_attention=attention(),
            norm1=Norm(d_model, rngs=rngs),
            norm2=Norm(d_model, rngs=rngs),
            feed_forward=FeedForward(
                d_model,
                config.feed_forward_size,
                Dropout(p_dropout=config.dropout_rate, rngs=rngs),
                rngs=rngs,
            ),
            dropout1=Dropout(p_dropout=config.dropout_rate, rngs=rngs),
            dropout2=Dropout(p_dropout=config.dropout_rate, rngs=rngs),
        )

    def decoder_layer() -> DecoderLayer:
        return DecoderLayer(
            self_attention=attention(),
            cross_attention=attention(),
            norm1=Norm(d_model, rngs=rngs),
            norm2=Norm(d_model, rngs=rngs),
            norm3=Norm(d_model, rngs=rngs),
            feed_forward=FeedForward(
                d_model,
                config.feed_forward_size,
                Dropout(p_dropout=config.dropout_rate, rngs=rngs),
                rngs=rngs,
            ),
            dropout1=Dropout(p_dropout=config.dropout_rate, rngs=rngs),
            dropout2=Dropout(p_dropout=config.dropout_rate, rngs=rngs),
            dropout3=Dropout(p_dropout=config.dropout_rate, rngs=rngs),
        )

    encoder = Encoder(
        embedder=Embedder(src_vocab_size, d_model, rngs=rngs),
        positional_encoding=PositionalEncoding(d_model, max_len),
        layers=[encoder_layer() for _ in range(config.num_encoder_layers)],
        norm=Norm(d_model, rngs=rngs),
        embedding_size=d_model,
    )
    decoder = Decoder(
        embedder=Embedder(tgt_vocab_size, d_model, rngs=rngs),
        positional_encoding=PositionalEncoding(d_model, max_len),
        layers=[decoder_layer() for _ in range(config.num_decoder_layers)],
        norm=Norm(d_model, rngs=rngs),
        embedding_size=d_model,
    )
    out = Linear(input_size=d_model, output_size=tgt_vocab_size, rngs=rngs)
    return Transformer(encoder, decoder, out)
