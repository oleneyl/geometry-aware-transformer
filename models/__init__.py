import tensorflow as tf
from models.modules.cnn import NMR_Infuse, SequentialCNNModel
from .sequence import RNNProteinModel, AttentionProteinModel, transformer_args
from models.modules.attention import DecoderLayer, VectorDecoder, EncoderLayer


def add_model_args(parser):
    transformer_args(parser)
    group = parser.add_argument_group('model')

    group.add_argument('--nmr_model', type=str, default='cnn')
    group.add_argument('--protein_model', type=str, default='gru')
    group.add_argument('--chemical_model', type=str, default='gru')
    group.add_argument('--protein_embedding_size', type=int, default=128)
    group.add_argument('--chemical_embedding_size', type=int, default=128)

    # Sequencial model control
    group.add_argument('--sequential_hidden_size', type=int, default=128)
    group.add_argument('--sequential_dropout', type=float, default=0.5)
    group.add_argument('--sequential_dense', type=int, default=64)

    # CNN model control
    group.add_argument('--cnn_filter_size', type=int, default=5)
    group.add_argument('--cnn_filter_number', type=int, default=32)
    group.add_argument('--cnn_hidden_layer', type=int, default=60)

    # Concat-model control
    group.add_argument('--concat_model', type=str, default='siamese')
    group.add_argument('--concat_hidden_layer_size', type=int, default=512)
    group.add_argument('--siamese_layer_size', type=int, default=32)
    group.add_argument('--concat_dropout', type=float, default=0.5)

    group.add_argument('--output_bias', type=float, default=0.0)
    
    group.add_argument('--modal_size', type=int, default=191)

    group.add_argument('--expansion_type', type=str, default='variable')


def create_sequence_model(args, model_type, vocab_size, export_level='end'):
    if model_type == 'gru':
        submodel = RNNProteinModel(args, vocab_size)
    elif model_type == 'att':
        submodel = AttentionProteinModel(args, vocab_size, export_level=export_level)
    elif model_type == 'cnn':
        submodel = SequentialCNNModel(args, vocab_size)
    return submodel


class BaseDTIModel(object):
    def __init__(self, args, export_level='end'):
        self._protein_encoded = tf.keras.Input(shape=[args.protein_sequence_length], dtype=tf.int32,
                                               name='protein_input')
        self._nmr_array = tf.keras.Input(shape=[args.nmr_array_size], dtype=tf.float32, name='nmr_input')
        self._smiles_encoded = tf.keras.Input(shape=[args.chemical_sequence_length], dtype=tf.int32, name='smile_input')
        self._modal_mask = tf.keras.Input(shape=[1, args.modal_size, args.modal_size], dtype=tf.float32, name='modal_mask')
        
        self.args = args

        self.protein_model = None
        self.chemical_model = None
        self.nmr_model = None


        with tf.name_scope('protein'):
            self.protein_model = create_sequence_model(args, 'att', args.protein_vocab_size, export_level=export_level)(self._protein_encoded)
        with tf.name_scope('chemical'):
            if args.nmr:
                self.nmr_model = NMR_Infuse(args, args.chemical_vocab_size)
                self.chemical_model = self.nmr_model(self._nmr_array, chemical_tensor=self._smiles_encoded, modal_mask=self._modal_mask)
            else:
                self.chemical_model = create_sequence_model(args, 'att', args.chemical_vocab_size, export_level=export_level)(self._smiles_encoded)

    def inputs(self):
        if self.args.nmr:
            return [self._protein_encoded, self._smiles_encoded, self._nmr_array, self._modal_mask]
        else:
            return [self._protein_encoded, self._smiles_encoded]

    def unsupervised_protein(self):
        pass

    def unsupervised_chemical(self):
        pass

    def predict_dti(self):
        return 1

    def inspect_model_output(self):
        return self.nmr_model, self.protein_model, self.chemical_model

    def create_keras_model(self):
        model_inputs = self.inputs()
        prediction = self.predict_dti()
        if not self.args.as_score:
            prediction = tf.keras.layers.Activation('sigmoid')(prediction)
        keras_model = tf.keras.Model(inputs=model_inputs, outputs=prediction)
        return keras_model


class Bias(tf.keras.layers.Layer):
    def __init__(self, bias):
        super(Bias, self).__init__(self)
        self.bias = bias

    def call(self, input):
        return tf.add(input, self.bias)


class InitialDTIModel(BaseDTIModel):
    def predict_dti(self):
        if self.args.nmr:
            if self.args.drop_smile:
                embedding = tf.keras.layers.concatenate([self.protein_model, self.nmr_model], 1)
            else:
                embedding = tf.keras.layers.concatenate([self.protein_model, self.chemical_model], 1)
        else:
            embedding = tf.keras.layers.concatenate([self.protein_model, self.chemical_model], 1)
        #                       self.chemical_model.get_output()], 1)
        embedding = tf.keras.layers.Dense(1024, activation='relu',
                                          name='concat_dense_1')(embedding)
        embedding = tf.keras.layers.BatchNormalization()(embedding)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding)
        embedding = tf.keras.layers.Dense(1024, activation='relu',
                                          name='concat_dense_2')(embedding)
        embedding = tf.keras.layers.BatchNormalization()(embedding)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding)
        embedding = tf.keras.layers.Dense(512, name='concat_dense_last', activation='relu')(embedding)
        embedding = tf.keras.layers.Dense(1, name='concat_dense_score', kernel_initializer='normal')(embedding)
        embedding = Bias(self.args.output_bias)(embedding)
        return embedding


class AttentiveDTIModel(BaseDTIModel):
    def __init__(self, args):
        super(AttentiveDTIModel, self).__init__(args, export_level='front')
        
    def predict_dti(self):
        decoder = VectorDecoder(self.args.transformer_num_layers,
                                self.args.transformer_model_dim,
                                self.args.transformer_num_heads,
                                self.args.transformer_hidden_dimension,
                                rate=self.args.transformer_dropout_rate)

        embedding, _ = decoder(self.protein_model, enc_output=self.chemical_model, look_ahead_mask=None, padding_mask=None)
        embedding = embedding[:, -1, :]
        embedding = tf.keras.layers.Dense(1024, activation='relu',
                                          name='concat_dense_1')(embedding)
        embedding = tf.keras.layers.BatchNormalization()(embedding)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding)
        embedding = tf.keras.layers.Dense(1024, activation='relu',
                                          name='concat_dense_2')(embedding)
        embedding = tf.keras.layers.BatchNormalization()(embedding)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding)
        embedding = tf.keras.layers.Dense(512, name='concat_dense_last', activation='relu')(embedding)
        embedding = tf.keras.layers.Dense(1, name='concat_dense_score', kernel_initializer='normal')(embedding)
        embedding = Bias(self.args.output_bias)(embedding)
        return embedding


class ConcatAttentionModel(BaseDTIModel):
    def predict_dti(self):
        protein_detect = self.protein_model.encoding
        chemical_detect = self.chemical_model.encoding

        x = tf.concat([protein_detect, chemical_detect], axis=1)
        # Cross - attention

        total_encoder = EncoderLayer(self.args.transformer_model_dim,
                                     self.args.transformer_num_heads,
                                     self.args.transformer_hidden_dimension,
                                     rate=self.args.transformer_dropout_rate)
        x = total_encoder(x, self.is_train, None)
        embedding = tf.keras.layers.Flatten()(x)
        embedding = tf.keras.layers.Dense(self.args.concat_hidden_layer_size, activation='relu',
                                          name='concat_dense_1')(embedding)
        embedding = tf.keras.layers.Dropout(self.args.concat_dropout)(embedding, training=self.is_train)
        embedding = tf.keras.layers.BatchNormalization()(embedding, training=self.is_train)
        dense_last = tf.keras.layers.Dense(1, name='concat_dense_last')
        embedding = dense_last(embedding)
        return embedding


def build_model(args):
    """
    Create output generation model from given placeholders
    """
    if args.concat_model == 'attention':
        return AttentiveDTIModel(args)
    else:
        return InitialDTIModel(args)


def get_model(args, saved_model=None):
    if saved_model:
        # Load model from saved_model path
        pass
    else:
        return build_model(args)
