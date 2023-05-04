#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import os
import logging
import string
import random
from datetime import datetime

from models.fit_moleculenet_task import get_moleculenet_task_dependent_arguments
from models.label_inference import InferenceAtomicNet

from training.trainer import Trainer
from training.metrics import Metrics
from training.data_container import DataContainer
from training.data_provider import DataProvider

from sacred import Experiment
import seml

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

ex = Experiment()
seml.setup_logger(ex)

# TensorFlow logging verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.get_logger().setLevel('WARN')
tf.autograph.set_verbosity(1)

ex.add_config('./config/config_H.yaml')

@ex.automain
def run(model_name,
        dataset, num_train, num_valid,
        data_seed, num_steps, learning_rate, ema_decay,
        decay_steps, warmup_steps, decay_rate, batch_size,
        evaluation_interval, save_interval, restart, targets,
        comment, logdir,
        chemical_sequence_length,
        chemical_vocab_size,
        transformer_model_dim,
        transformer_num_heads,
        transformer_hidden_dimension,
        transformer_num_layers,
        expansion_type,
        basis_distance,
        cg_size,
        angular,
        _config):

    print(_config)
    # Used for creating a "unique" id for a run (almost impossible to generate the same twice)
    def id_generator(size=8, chars=string.ascii_uppercase + string.ascii_lowercase + string.digits):
        return ''.join(random.SystemRandom().choice(chars) for _ in range(size))

    # Create directories
    # A unique directory name is created for this run based on the input
    if restart is None:
        directory = (
                logdir + "/" + datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + id_generator()
                + "_" + os.path.basename(dataset)
                + f"_lr{learning_rate:.2e}"
                + f"_dec{decay_steps:.2e}"
                + f"_tfmdim{transformer_model_dim:.2e}"
                + f"_tfnhead{transformer_num_heads:.2e}"
                + f"_tfhdim{transformer_hidden_dimension:.2e}"
                + f"_tfnlayer{transformer_num_layers:.2e}"
                + f"_expt-{expansion_type}"
                + "_" + '-'.join(targets)
                + "_" + comment
        )
    else:
        directory = restart

    logging.info(f"Directory: {directory}")

    logging.info("Create directories")
    if not os.path.exists(directory):
        os.makedirs(directory)
    best_dir = os.path.join(directory, 'best')
    if not os.path.exists(best_dir):
        os.makedirs(best_dir)
    log_dir = os.path.join(directory, 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    best_loss_file = os.path.join(best_dir, 'best_loss.npz')
    best_ckpt_file = os.path.join(best_dir, 'ckpt')
    step_ckpt_folder = log_dir

    # Initialize summary writer
    summary_writer = tf.summary.create_file_writer(log_dir)

    train = {}
    validation = {}

    # Initialize metrics
    train['metrics'] = Metrics('train', targets, ex)
    validation['metrics'] = Metrics('val', targets, ex)

    with summary_writer.as_default():
        logging.info("Load dataset")
        data_container = DataContainer(dataset, cutoff=4, target_keys=targets, angular=angular, cg_size=cg_size, basis_distance=basis_distance)

        # Initialize DataProvider (splits dataset into 3 sets based on data_seed and provides tf.datasets)
        data_provider = DataProvider(data_container, num_train, num_valid, batch_size,
                                     seed=data_seed, randomized=True, angular=angular, cg_size=cg_size)

        # Initialize datasets
        train['dataset'] = data_provider.get_dataset('train').prefetch(tf.data.experimental.AUTOTUNE)
        validation['dataset'] = data_provider.get_dataset('val').prefetch(tf.data.experimental.AUTOTUNE)

        train['dataset_iter'] = iter(train['dataset'])    
        validation['dataset_iter'] = iter(validation['dataset'])

        # Create model
        logging.info("Initialize model")
        model_fitting_information = get_moleculenet_task_dependent_arguments('QM9Single')

        nmr_interaction = InferenceAtomicNet(chemical_sequence_length,
                                        chemical_vocab_size,
                                        transformer_model_dim,
                                        transformer_num_heads,
                                        transformer_hidden_dimension,
                                        transformer_num_layers,
                                        expansion_type,
                                        basis_distance=basis_distance,
                                        label_count=model_fitting_information['label_count'],
                                        is_regression=model_fitting_information['is_regression'],
                                        use_attention_scale=True, 
                                        use_atom_embedding=True,
                                        use_parallel_mlp=True)

        model = nmr_interaction.create_keras_model()

        logging.info("Prepare training")
        # Save/load best recorded loss (only the best model is saved)
        if os.path.isfile(best_loss_file):
            loss_file = np.load(best_loss_file)
            metrics_best = {k: v.item() for k, v in loss_file.items()}
        else:
            metrics_best = {k: np.inf for k in validation['metrics'].keys()}
            metrics_best['step'] = 0
            np.savez(best_loss_file, **metrics_best)

        # Initialize trainer
        trainer = Trainer(model, learning_rate, warmup_steps,
                          decay_steps, decay_rate,
                          ema_decay=ema_decay, max_grad_norm=1000)

        # Set up checkpointing
        ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=trainer.optimizer, model=model)
        manager = tf.train.CheckpointManager(ckpt, step_ckpt_folder, max_to_keep=3)

        # Restore latest checkpoint
        ckpt_restored = tf.train.latest_checkpoint(log_dir)
        if ckpt_restored is not None:
            ckpt.restore(ckpt_restored)

        if ex is not None:
            ex.current_run.info = {'directory': directory}

        # Training loop
        logging.info("Start training")
        steps_per_epoch = int(np.ceil(num_train / batch_size))

        if ckpt_restored is not None:
            step_init = ckpt.step.numpy()
        else:
            step_init = 1
        for step in range(step_init, num_steps + 1):
            # Update step number
            ckpt.step.assign(step)
            tf.summary.experimental.set_step(step)

            # Perform training step
            trainer.train_on_batch(train['dataset_iter'], train['metrics'], from_tfrecord=False)

            # Save progress
            if (step % save_interval == 0):
                manager.save()

            # Check performance on the validation set
            if (step % evaluation_interval == 0):

                # Save backup variables and load averaged variables
                trainer.save_variable_backups()
                trainer.load_averaged_variables()
                # Compute averages
                for i in range(int(np.ceil(num_valid / batch_size))):
                    trainer.test_on_batch(validation['dataset_iter'], validation['metrics'])

                # Update and save best result
                if validation['metrics'].mean_mae < metrics_best['mean_mae_val']:
                    metrics_best['step'] = step
                    metrics_best.update(validation['metrics'].result())

                    np.savez(best_loss_file, **metrics_best)
                    model.save_weights(os.path.join(best_ckpt_file, str(step // 1e5)+"00k" ))

                for key, val in metrics_best.items():
                    if key != 'step':
                        tf.summary.scalar(key + '_best', val)

                epoch = step // steps_per_epoch
                logging.info(
                    f"{step}/{num_steps} (epoch {epoch+1}): "
                    f"Loss: train={train['metrics'].loss:.6f}, val={validation['metrics'].loss:.6f}; "
                    f"logMAE: train={train['metrics'].mean_log_mae:.6f}, "
                    f"val={validation['metrics'].mean_log_mae:.6f}")

                train['metrics'].write()
                validation['metrics'].write()

                train['metrics'].reset_states()
                validation['metrics'].reset_states()

                # Restore backup variables
                trainer.restore_variable_backups()

    return {key + '_best': val for key, val in metrics_best.items()}
