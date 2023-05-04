from collections import OrderedDict
import numpy as np
import tensorflow as tf


class DataProvider:
    def __init__(self, data_container, ntrain, nvalid, batch_size=1,
                 seed=None, randomized=False, angular=False, is_contain_force=False, chemical_sequence_length=32, cg_size=1, map_everything_to_train=False):
        self.data_container = data_container
        self._ndata = len(data_container)

        if map_everything_to_train:
            self.nsamples = {'train': len(data_container), 'val': 0, 'test': 0}
        else:
            self.nsamples = {'train': ntrain, 'val': nvalid, 'test': len(data_container) - ntrain - nvalid}
        self.batch_size = batch_size
        self.angular = angular
        self.cg_size = cg_size
        self.is_contain_force = is_contain_force
        self.chemical_sequence_length = chemical_sequence_length
        self.randomized = randomized

        # Random state parameter, such that random operations are reproducible if wanted
        self._random_state = np.random.RandomState(seed=seed)

        all_idx = np.arange(len(self.data_container))

        if randomized:
            # Shuffle indices
            all_idx = self._random_state.permutation(all_idx)

        if map_everything_to_train:
            self.idx = {'train': all_idx,
                    'val': [],
                    'test': []}
        else:
            # Store indices of training, validation and test data
            self.idx = {'train': all_idx[0:ntrain],
                        'val': all_idx[ntrain:ntrain+nvalid],
                        'test': all_idx[ntrain+nvalid:]}

        # Index for retrieving batches
        self.idx_in_epoch = {'train': 0, 'val': 0, 'test': 0}

        # dtypes of dataset values
        self._set_dtype_and_shape_of_inputs(data_container, chemical_sequence_length=self.chemical_sequence_length)

    def _set_dtype_and_shape_of_inputs(self, data_container, chemical_sequence_length=32):
        self.dtypes_input = OrderedDict()
        self.shapes_input = {}
        defined_data_types = [
            ('atom_type', tf.int32, [chemical_sequence_length]),
            ('orbit_coeff', tf.float32, [chemical_sequence_length, chemical_sequence_length]),
            ('distance', tf.float32, [chemical_sequence_length, chemical_sequence_length]),
            ('extract_matrix', tf.float32, [chemical_sequence_length, chemical_sequence_length]),
            ('pad_mask', tf.float32, [1, chemical_sequence_length] if self.angular else [1, 1, chemical_sequence_length]),
            ('output_mask', tf.float32, [chemical_sequence_length]),
        ]

        if self.angular:
            cg_dim = (1 + self.cg_size) * (1 + self.cg_size)
            defined_data_types.append(
                ('angular', tf.float32, [chemical_sequence_length, chemical_sequence_length, cg_dim, cg_dim])
            )
        if self.is_contain_force:
            defined_data_types.append(
                ('F', tf.float32, [chemical_sequence_length, 3])
            )
            defined_data_types.append(
                ('R', tf.float32, [chemical_sequence_length, 3])
            )

        for data_name, data_type, data_shape in defined_data_types:
            self.dtypes_input[data_name] = data_type
            self.shapes_input[data_name] = [None] + data_shape
        
        self.dtype_target = tf.float32
        self.shape_target = [None, len(data_container.target_keys)]

    def shuffle_train(self):
        """Shuffle the training data"""
        self.idx['train'] = self._random_state.permutation(self.idx['train'])

    def get_batch_idx(self, split, to_be_tfrecord=False):
        """Return the indices for a batch of samples from the specified set"""
        start = self.idx_in_epoch[split]

        # Is epoch finished?
        if self.idx_in_epoch[split] == self.nsamples[split]:
            start = 0
            self.idx_in_epoch[split] = 0

        # shuffle training set at start of epoch
        if not to_be_tfrecord:
            if start == 0 and split == 'train' and self.randomized:
                self.shuffle_train()

        # Set end of batch
        self.idx_in_epoch[split] += self.batch_size
        if self.idx_in_epoch[split] > self.nsamples[split]:
            self.idx_in_epoch[split] = self.nsamples[split]
        end = self.idx_in_epoch[split]

        return self.idx[split][start:end]

    def idx_to_data(self, idx, return_flattened=False):
        """Convert a batch of indices to a batch of data"""
        batch = self.data_container[idx]

        if return_flattened:
            inputs_targets = []
            for key, dtype in self.dtypes_input.items():
                inputs_targets.append(tf.constant(batch[key], dtype=dtype))
            inputs_targets.append(tf.constant(batch['targets'], dtype=tf.float32))
            return inputs_targets
        else:
            inputs = {}
            for key, dtype in self.dtypes_input.items():
                inputs[key] = tf.constant(batch[key], dtype=dtype)
            targets = tf.constant(batch['targets'], dtype=tf.float32)
            return (inputs, targets)

    def get_dataset(self, split, to_be_tfrecord=False):
        """Get a generator-based tf.dataset"""
        def generator():
            started = False
            while True:
                idx = self.get_batch_idx(split, to_be_tfrecord=to_be_tfrecord)
                if to_be_tfrecord:
                    if not started:
                        started = idx
                    else:
                        if started == idx:
                            break
                yield self.idx_to_data(idx)
        return tf.data.Dataset.from_generator(
                generator,
                output_types=(dict(self.dtypes_input), self.dtype_target),
                output_shapes=(self.shapes_input, self.shape_target))

    def get_idx_dataset(self, split):
        """Get a generator-based tf.dataset returning just the indices"""
        def generator():
            while True:
                batch_idx = self.get_batch_idx(split)
                yield tf.constant(batch_idx, dtype=tf.int32)
        return tf.data.Dataset.from_generator(
                generator,
                output_types=tf.int32,
                output_shapes=[None])

    def idx_to_data_tf(self, idx):
        """Convert a batch of indices to a batch of data from TensorFlow"""
        dtypes_flattened = list(self.dtypes_input.values())
        dtypes_flattened.append(self.dtype_target)

        inputs_targets = tf.py_function(lambda idx: self.idx_to_data(idx.numpy(), return_flattened=True),
                                        inp=[idx], Tout=dtypes_flattened)

        inputs = {}
        for i, key in enumerate(self.dtypes_input.keys()):
            inputs[key] = inputs_targets[i]
            inputs[key].set_shape(self.shapes_input[key])
        targets = inputs_targets[-1]
        targets.set_shape(self.shape_target)
        return (inputs, targets)

    @classmethod
    def _get_feature_description(cls, chemical_sequence_length):
        return {
            'atom_type': tf.io.FixedLenFeature([chemical_sequence_length], tf.int64),
            'orbit_coeff': tf.io.FixedLenFeature([chemical_sequence_length, chemical_sequence_length], tf.float32),
            'distance': tf.io.FixedLenFeature([chemical_sequence_length, chemical_sequence_length], tf.float32),
            'extract_matrix': tf.io.FixedLenFeature([chemical_sequence_length, chemical_sequence_length], tf.float32),
            'pad_mask': tf.io.FixedLenFeature([1, 1, chemical_sequence_length], tf.float32),
            'output_mask': tf.io.FixedLenFeature([chemical_sequence_length], tf.float32),
            'target': tf.io.FixedLenFeature([1], tf.float32),
        }

    @classmethod
    def _get_parse_function(cls, feature_description):
        def _parse_function(serialized_example):
            # Parse the input `tf.train.Example` proto using the dictionary above.
            features = tf.io.parse_single_example(serialized_example, feature_description)
            return features
        
        return _parse_function

    @classmethod
    def from_prebuilt_dataset(cls, file_path, chemical_sequence_length, batch_size):
        feature_description = cls._get_feature_description(chemical_sequence_length)
        parse_function = cls._get_parse_function(feature_description)
        
        return tf.data.TFRecordDataset(file_path).map(parse_function).batch(batch_size)
