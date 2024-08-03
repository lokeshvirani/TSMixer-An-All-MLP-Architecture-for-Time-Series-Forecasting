

"""Implementation of TSMixer with Reversible Instance Normalization."""

from models.rev_in import RevNorm
from models.tsmixer import res_block
import tensorflow as tf
from tensorflow.keras import layers


def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
  """Build TSMixer with Reversible Instance Normalization model."""

  inputs = tf.keras.Input(shape=input_shape)
  x = inputs  # [Batch, Input Length, Channel]
  rev_norm = RevNorm(axis=-2)
  x = rev_norm(x, 'norm')
  for _ in range(n_block):
    x = res_block(x, norm_type, activation, dropout, ff_dim)

  if target_slice:
    x = x[:, :, target_slice]

  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
  outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])
  outputs = rev_norm(outputs, 'denorm', target_slice)
  return tf.keras.Model(inputs, outputs)
