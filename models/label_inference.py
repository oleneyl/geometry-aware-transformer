import tensorflow as tf
import numpy as np
from .modules.atomic import AtomicLayer


class InferenceAtomicNet(object):
    def __init__(self, 
                 chemical_sequence_length,
                 chemical_vocab_size,
                 transformer_model_dim,
                 transformer_num_heads,
                 transformer_hidden_dimension,
                 transformer_num_layers,
                 expansion_type,
                 label_count=12,
                 basis_distance=30,
                 is_regression=False,
                 use_attention_scale=False, 
                 use_atom_embedding=False,
                 use_parallel_mlp=False):

        self._atom_type = tf.keras.Input(shape=[chemical_sequence_length], dtype=tf.int32, name='atom_type')
        self._orbit_coeff = tf.keras.Input(shape=[chemical_sequence_length, chemical_sequence_length], dtype=tf.float32, name='orbit_coeff')
        self._distance = tf.keras.Input(shape=[chemical_sequence_length, chemical_sequence_length], dtype=tf.float32, name='distance')
        self._extract_matrix = tf.keras.Input(shape=[chemical_sequence_length, chemical_sequence_length], dtype=tf.float32, name='extract_matrix')
        self._pad_mask = tf.keras.Input(shape=[1, 1, chemical_sequence_length], dtype=tf.float32,
                                          name='pad_mask')
        self._output_mask = tf.keras.Input(shape=[chemical_sequence_length], dtype=tf.float32, name='output_mask')
        #self._position = tf.keras.Input(shape=[chemical_sequence_length, 3], dtype=tf.float32, name='R')

        self.label_count = label_count
        self.is_regression = is_regression

        print(use_attention_scale, use_atom_embedding, use_parallel_mlp)

        self.embedding = tf.keras.layers.Embedding(chemical_vocab_size, transformer_model_dim, embeddings_initializer='normal')
        self.atomic_layer = [
            AtomicLayer(transformer_model_dim,
                        transformer_num_heads,
                        transformer_hidden_dimension,
                        basis_distance=basis_distance,
                        expansion_type=expansion_type,
                        use_attention_scale=use_attention_scale, 
                        use_atom_embedding=use_atom_embedding,
                        use_parallel_mlp=use_parallel_mlp
            ) for i in range(transformer_num_layers)
        ]
        self.layer_num = transformer_num_layers

        self.mlp_1 = tf.keras.layers.Dense(64, activation='elu')
        self.output_dense = tf.keras.layers.Dense(self.label_count)

    def inputs(self):
        return [self._atom_type,
                self._orbit_coeff,
                self._distance,
                #self._position,
                self._extract_matrix,
                self._pad_mask,
                self._output_mask]


    def predict(self):
        orbit_state = self.embedding(self._atom_type)

        for i in range(self.layer_num):
            orbit_state = self.atomic_layer[i]([orbit_state, self._distance, self._atom_type], mask=self._pad_mask)

        gather_state = orbit_state * tf.expand_dims(self._output_mask, axis=-1)
        gather_state = tf.reduce_sum(gather_state, axis=1)
        inference_output = self.mlp_1(gather_state)
        inference_output = self.output_dense(inference_output) - 6.529990

        return inference_output

    def create_keras_model(self):
        model_inputs = self.inputs()
        prediction = self.predict()

        keras_model = tf.keras.Model(inputs=model_inputs, outputs=prediction)
        return keras_model