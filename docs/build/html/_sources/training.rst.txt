.. _training:

========
Training
========

This section covers all aspects of training MolE models, from basic fine-tuning to advanced pretraining strategies.

Training Overview
=================

MolE employs a two-stage training approach:

1. **Self-supervised pretraining** on ~842 million molecules
2. **Multi-task fine-tuning** on biological datasets

Fine-tuning Guide
=================

Basic Fine-tuning
-----------------

Use the ``mole_train`` command with the ``finetune`` model configuration:

.. code-block:: bash

   mole_train model=finetune \
     data_file='path/to/training_data.csv' \
     checkpoint_path='path/to/pretrained_model.ckpt' \
     dropout=0.1 \
     lr=1.0e-06 \
     task=regression \
     num_tasks=1

Parameters
----------

Required Parameters
~~~~~~~~~~~~~~~~~~~

- **data_file** (string): Path to training data file
- **checkpoint_path** (string): Path to pretrained model, or ``null`` for random initialization
- **dropout** (float): Dropout rate for prediction head (0.0-1.0)
- **lr** (float): Learning rate for fine-tuning

Optional Parameters
~~~~~~~~~~~~~~~~~~~

- **task** (string): ``regression`` or ``classification`` (default: regression)
- **num_tasks** (int): Number of prediction tasks (default: 1)
- **model.name** (string): Name for the fine-tuned model
- **model.hyperparameters.datamodule.validation_data** (string): Path to validation data

Configuration Management
========================

Viewing Configuration
---------------------

Before training, view the complete configuration:

.. code-block:: bash

   mole_train model=finetune --cfg job --resolve

This prints the full configuration without starting training.

Custom Configuration
--------------------

Override specific parameters:

.. code-block:: bash

   mole_train model=finetune \
     data_file='data.csv' \
     model.hyperparameters.optimizer.lr=5e-5 \
     model.hyperparameters.trainer.max_epochs=100

Training Examples
=================

Regression Task
---------------

.. code-block:: bash

   mole_train model=finetune \
     data_file='data/TDC_Half_Life_Obach_train_seed0.parquet' \
     checkpoint_path=null \
     dropout=0.1 \
     lr=1.0e-06 \
     task=regression \
     num_tasks=1 \
     model.name='MolE_Half_Life_Regression' \
     model.hyperparameters.datamodule.validation_data='data/TDC_Half_Life_Obach_valid_seed0.parquet'

Classification Task
-------------------

.. code-block:: bash

   mole_train model=finetune \
     data_file='data/TDC_HIA_Hou_train_seed0.csv' \
     checkpoint_path=null \
     dropout=0.1 \
     lr=1.0e-06 \
     task=classification \
     num_tasks=1 \
     model.name='MolE_HIA_Classification' \
     model.hyperparameters.datamodule.validation_data='data/TDC_HIA_Hou_valid_seed0.csv'

Advanced Training
=================

Multi-task Learning
-------------------

Train on multiple properties simultaneously:

.. code-block:: bash

   mole_train model=finetune \
     data_file='data/multi_property_dataset.csv' \
     num_tasks=5 \
     task=regression

Custom Learning Schedules
--------------------------

Use custom learning rate schedules and optimizers through configuration overrides:

.. code-block:: bash

   mole_train model=finetune \
     data_file='data.csv' \
     model.hyperparameters.optimizer.weight_decay=0.01 \
     model.hyperparameters.scheduler.warmup_steps=1000

Monitoring Training
===================

Training logs and checkpoints are saved to the output directory. Monitor progress using:

- Loss curves and metrics in training logs
- Model checkpoints for resuming training
- Validation metrics for early stopping

For more details on the underlying training algorithms and pretraining strategies, see the :doc:`pretraining` documentation. 