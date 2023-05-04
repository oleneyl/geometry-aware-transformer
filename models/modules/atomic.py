import tensorflow as tf
import numpy as np
from .attention import point_wise_feed_forward_network
from .bessel import SphericalBasisLayer
import sys


def shifted_softplus(x):
    x = tf.keras.activations.softplus(x)
    return x - 0.69314718056

def scaled_weighted_dot_product_attention(q, k, v, weight, mask, attention_scale=None):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    The mask has different shapes depending on its type(padding or look ahead)
    but it must be broadcastable for addition.

    Args:
      q: query shape == (..., seq_len_q, depth)
      k: key shape == (..., seq_len_k, depth)
      v: value shape == (..., seq_len_v, depth_v)
      mask: Float tensor with shape broadcastable
            to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
      output, attention_weights
    """
    q = tf.expand_dims(q, -2)  # (batch, head, seq_len, 1, depth)
    k = tf.expand_dims(k, -3)  # (batch, head, 1, seq_len, depth)

    r = weight
    dk = tf.cast(tf.shape(k)[-1], tf.float32)

    scaled_attention_logits = tf.reduce_sum(q * k * r, axis=-1) / tf.math.sqrt(dk)

    USE_SOFTMAX = False
    # add the mask to the scaled tensor.

    if mask is not None:
        if USE_SOFTMAX:
            scaled_attention_logits += (mask * -1e9)
        else:
            scaled_attention_logits = scaled_attention_logits * (1 - mask)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = scaled_attention_logits

    if USE_SOFTMAX:
        attention_weights = tf.nn.softmax(attention_weights, axis=-1)  # (..., seq_len_q, seq_len_k)

    if attention_scale is not None:
        attention_weights_mean = tf.reduce_sum(attention_weights, axis=-1, keepdims=True)
        mask_mean = tf.reduce_mean(1 - mask, axis=-1, keepdims=True)

        seq_len = attention_weights.shape[-1]
        attention_weights_mean = attention_weights_mean * (1 - mask) / seq_len / mask_mean

        attention_weights = (attention_weights - attention_weights_mean) * (1 + attention_scale) + attention_weights_mean

    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class AtomicMultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, expansion_type='variable', basis_distance=30, 
                 use_attention_scale=False, 
                 use_atom_embedding=False):
        super(AtomicMultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.basis_distance = basis_distance

        assert d_model % self.num_heads == 0
        self.expansion_type = expansion_type

        self.depth = d_model // self.num_heads

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

        # Radial kernel
        self.radial_expansion_1 = tf.keras.layers.Dense(self.depth, activation='elu')
        self.radial_expansion_2 = tf.keras.layers.Dense(self.depth * self.num_heads)

        self.radial_expansion_sf = tf.keras.layers.Dense(self.depth * self.num_heads, activation='elu')

        # Polynomial kernel
        self.polynomial_expansion = tf.keras.layers.Dense(self.depth)

        self.schnet_1 = tf.keras.layers.Dense(64, activation=shifted_softplus)
        self.schnet_2 = tf.keras.layers.Dense(self.depth * self.num_heads, activation=shifted_softplus)

        if expansion_type == 'bessel':
            self.bessel = SphericalBasisLayer(4, 32, 50)

        ### Attention Scale
        self.use_attention_scale = use_attention_scale
        if self.use_attention_scale:
            self.attn_scale_weight = tf.Variable(initial_value=0.1, dtype=tf.float32, trainable=True)
        else:
            self.attn_scale_weight = None

        ### Atom Embedding
        self.use_atom_embedding = use_atom_embedding
        if self.use_atom_embedding:
            self.atom_embedding = tf.keras.layers.Embedding(200, 64, embeddings_initializer='normal')

    def expand_to_radial(self, distance, atom_type, expansion_type='variable'):
        '''

        :param distance: distance matrix, [seq_len, seq_len]
        :param expansion_type: string, expansion type.
        :return: tensor, [seq_len, seq_len, self.depth]
        '''

        if expansion_type == 'physnet':
            exp_distance = tf.expand_dims(distance, axis=-1)
            divident = [np.arange(0, 30, 0.1).tolist()]
            rbf = exp_distance / divident
            rbf = (1 - rbf * rbf * rbf * 6 + 15 * rbf * rbf * rbf * rbf - 10 * rbf * rbf * rbf * rbf * rbf)
            rbf = tf.maximum(rbf, 0.0)
            exp_distance = self.radial_expansion_1(rbf)
            exp_distance = self.radial_expansion_2(exp_distance)

            return exp_distance

        if expansion_type == 'schnet':
            exp_distance = tf.expand_dims(distance, axis=-1)
            dist_delta = [np.arange(0, self.basis_distance, 0.1).tolist()]
            exp_distance = exp_distance - dist_delta
            exp_distance = exp_distance * exp_distance * (-10)
            exp_distance = tf.exp(exp_distance)

            if self.use_atom_embedding:
                atom_embedding = self.atom_embedding(atom_type)

                seq_len = tf.shape(exp_distance)[1]

                a_embedding = tf.repeat(
                    tf.expand_dims(atom_embedding, axis=1),
                    repeats=seq_len,
                    axis=1
                )
                b_embedding = tf.repeat(
                    tf.expand_dims(atom_embedding, axis=2),
                    repeats=seq_len,
                    axis=2
                )

                exp_distance = tf.concat([
                    exp_distance, a_embedding + b_embedding
                ], 3)

            exp_distance = self.schnet_1(exp_distance)
            exp_distance = self.schnet_2(exp_distance)

            exp_distance = exp_distance
            return exp_distance

        if expansion_type == 'bessel':
            exp_distance = tf.expand_dims(distance, axis=-1)
            exp_distance = self.bessel(exp_distance)
            exp_distance = tf.squeeze(exp_distance, axis=-1)
            exp_distance = tf.transpose(exp_distance, [0, 2, 3, 1])
            exp_distance = self.schnet_1(exp_distance)
            exp_distance = self.schnet_2(exp_distance)

            return exp_distance

        if expansion_type == 'polynomial2':
            inv_dist = tf.minimum(1 / distance, 3) / 3
            poly_dist = tf.stack([inv_dist, inv_dist * inv_dist, inv_dist * inv_dist * inv_dist], axis=-1)
            poly_dist = self.polynomial_expansion(poly_dist)

            exp_distance = tf.expand_dims(distance, axis=-1)
            exponent_expansion = tf.matmul(exp_distance * -1, [np.arange(1, 3.001, 2.0 / (self.depth - 1)).tolist()])
            exponent_expansion = tf.exp(exponent_expansion) * poly_dist

            exponent_expansion = self.radial_expansion_sf(exponent_expansion)
            exponent_expansion = self.radial_expansion_2(exponent_expansion)

            return exponent_expansion

        if expansion_type == 'polynomial':
            inv_dist = tf.minimum(1 / distance, 3) / 3
            poly_dist = tf.stack([inv_dist, inv_dist * inv_dist, inv_dist * inv_dist * inv_dist], axis=-1)
            poly_dist = self.polynomial_expansion(poly_dist)

            exp_distance = tf.expand_dims(distance, axis=-1)
            exponent_expansion = tf.matmul(exp_distance * -1, [np.arange(1, 3.001, 2.0 / (self.depth - 1)).tolist()])
            exponent_expansion = tf.exp(exponent_expansion) * poly_dist

            exponent_expansion = self.radial_expansion_1(exponent_expansion)
            exponent_expansion = self.radial_expansion_2(exponent_expansion)

            return exponent_expansion

        if expansion_type == 'variable':
            distance = tf.expand_dims(distance, axis=-1)
            exponent_expansion = tf.matmul(distance * -1, [np.arange(0.1, 3.001, 2.9 / (self.depth - 1)).tolist()])
            exponent_expansion = tf.minimum(0.3 / distance, 3) * tf.exp(exponent_expansion)
            exponent_expansion = self.radial_expansion_1(exponent_expansion)
            exponent_expansion = self.radial_expansion_2(exponent_expansion)

            return exponent_expansion

        if expansion_type == 'fixed':
            distance = tf.expand_dims(distance, axis=-1)
            exponent_expansion = tf.matmul(distance * -1, [np.arange(0.1, 3.001, 2.9 / (self.depth - 1)).tolist()])
            return tf.minimum(0.5 / distance, 2) * tf.exp(exponent_expansion)

        if expansion_type == 'linear':
            distance = tf.expand_dims(distance, axis=-1)
            exponent_expansion = tf.matmul(distance * -1, [[1.0 for i in range(self.depth * self.num_heads)]])
            return tf.minimum(0.5 / distance, 2) * tf.exp(exponent_expansion)


    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, vkq_weight, mask=None, training=None):
        v, k, q, weight, atom_type = vkq_weight
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        shifted_distance = self.expand_to_radial(weight, atom_type, expansion_type=self.expansion_type)


        shifted_distance = tf.reshape(shifted_distance, (batch_size, tf.shape(shifted_distance)[1], tf.shape(shifted_distance)[2], self.num_heads, self.depth))
        shifted_distance = tf.transpose(shifted_distance, [0, 3, 1, 2, 4])

        scaled_attention, attention_weights = scaled_weighted_dot_product_attention(
            q, k, v, shifted_distance, mask, self.attn_scale_weight)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights

class InteracionBlock(tf.keras.layers.Layer):
    def __init__(self, d_model):
        super(InteracionBlock, self).__init__()

        self.schnet_1 = tf.keras.layers.Dense(64, activation=shifted_softplus)
        self.schnet_2 = tf.keras.layers.Dense(d_model, activation=shifted_softplus)

        self.atom_wise_1 = tf.keras.layers.Dense(d_model)
        self.atom_wise_2 = tf.keras.layers.Dense(d_model, activation=shifted_softplus)
        self.atom_wise_3 = tf.keras.layers.Dense(d_model)

    def expand_to_radial(self, distance):
        '''

        :param distance: distance matrix, [seq_len, seq_len]
        :param expansion_type: string, expansion type.
        :return: tensor, [seq_len, seq_len, self.depth]
        '''
        exp_distance = tf.expand_dims(distance, axis=-1)
        dist_delta = [np.arange(0, 30, 0.1).tolist()]
        exp_distance = exp_distance - dist_delta
        exp_distance = exp_distance * exp_distance * (-10)
        exp_distance = tf.exp(exp_distance)
        exp_distance = self.schnet_1(exp_distance)
        exp_distance = self.schnet_2(exp_distance)
        return exp_distance

    def call(self, v_dist, mask=None, training=None):
        v, distance = v_dist   # v, [BN, seq_len, model_dim]

        mask = tf.squeeze(mask, axis=1)
        mask = tf.squeeze(mask, axis=1)
        mask = tf.expand_dims(mask, axis=-1)

        v = v * mask
        v = self.atom_wise_1(v)
        v = v * mask
        shifted_distance = self.expand_to_radial(distance)
        #  [BN, seq_len, seq_len, model_dim], [BN, seq_len, model_dim]
        v = tf.expand_dims(v, axis=1)
        v = v * shifted_distance # [BN, seq_len(1), seq_len, model_dim]
        v = tf.reduce_sum(v, axis=2) # [BN, seq_len(1), seq_len -> 0, model_dim]

        v = v * mask
        v = self.atom_wise_2(v)
        v = v * mask
        v = self.atom_wise_3(v)
        v = v * mask
        return v

class AtomicLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.0, basis_distance=30, expansion_type='variable',
                 use_attention_scale=False, 
                 use_atom_embedding=False,
                 use_parallel_mlp=False):
        super(AtomicLayer, self).__init__()

        self.creator = AtomicMultiHeadAttention(d_model, num_heads, expansion_type=expansion_type, basis_distance=basis_distance,
            use_attention_scale=use_attention_scale,
            use_atom_embedding=use_atom_embedding
        )
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        if not use_parallel_mlp:
            self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

        self._save_recent_coeff = None
        print(f"use_attention_scale {use_attention_scale} | use_atom_embedding {use_atom_embedding} | use_parallel_mlp {use_parallel_mlp}")
        self.use_parallel_mlp = use_parallel_mlp


    def call(self, state_weight, mask=None, training=None):
        orbital_state, overlap_weight, atom_type = state_weight

        # Creator
        attn_output, tmp_attn = self.creator([orbital_state, orbital_state, orbital_state, overlap_weight, atom_type], mask=mask, training=training)
        self._save_recent_coeff = tmp_attn

        if self.use_parallel_mlp:
            attn_output = self.dropout1(attn_output, training=training)
            ffn_output = self.ffn(orbital_state)  # (batch_size, input_seq_len, d_model)
            ffn_output = self.dropout2(ffn_output, training=training)

            out1 = self.layernorm1(orbital_state + attn_output + ffn_output)  # (batch_size, input_seq_len, d_model)

            return out1
        else:
            attn_output = self.dropout1(attn_output, training=training)
            out1 = self.layernorm1(orbital_state + attn_output)  # (batch_size, input_seq_len, d_model)
            ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
            ffn_output = self.dropout2(ffn_output, training=training)
            new_state = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
            return new_state
