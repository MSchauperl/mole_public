.. _quickstart:

==========
Quickstart
==========

This guide will get you up and running with MolE quickly for common use cases.

Training a Model
================

Fine-tuning MolE
----------------

To fine-tune MolE on your own dataset:

.. code-block:: bash

   # Regression example
   mole_train model=finetune \
     data_file='data/your_training_data.parquet' \
     checkpoint_path=null \
     dropout=0.1 \
     lr=1.0e-06 \
     task=regression \
     num_tasks=1 \
     model.name='MolE_Finetune_Regression'

   # Classification example  
   mole_train model=finetune \
     data_file='data/your_training_data.csv' \
     checkpoint_path=null \
     dropout=0.1 \
     lr=1.0e-06 \
     task=classification \
     num_tasks=1 \
     model.name='MolE_Finetune_Classification'

Data Format
-----------

Your training data should be a CSV/Parquet file with at least a **smiles** column and property columns:

.. list-table::
   :header-rows: 1

   * - smiles
     - Property1
     - Property2
   * - CCC
     - 301
     - 283
   * - c1ccccc1
     - 192
     - 327

Making Predictions
==================

Python API
----------

.. code-block:: python

   from mole import mole_predict
   import pandas as pd

   # Define molecules
   smiles = ['CCC', 'CCCCCC', 'CC', 'CCCCC']

   # Make predictions
   predictions = mole_predict.predict(
       smiles=smiles, 
       task='regression', 
       num_tasks=1, 
       pretrained_model='path/to/checkpoint.ckpt',
       batch_size=32, 
       num_workers=4
   )

   # Create results dataframe
   df = pd.DataFrame(predictions)
   df.insert(0, 'smiles', smiles)
   print(df.head())

Command Line
------------

.. code-block:: bash

   mole_predict \
     --smiles "CCC c1ccccc1" \
     --task regression \
     --num_tasks 2 \
     --pretrained_model path/to/checkpoint.ckpt

Computing Embeddings
====================

Extract molecular embeddings for similarity search or fingerprinting:

.. code-block:: python

   from mole import mole_predict

   smiles = ['CCC', 'CCCCCC', 'CC', 'CCCCC']

   embeddings = mole_predict.encode(
       smiles=smiles, 
       pretrained_model='path/to/checkpoint.ckpt',
       batch_size=32, 
       num_workers=4
   )
   
   print(f"Embeddings shape: {embeddings.shape}")

Configuration
=============

View complete configuration before training:

.. code-block:: bash

   mole_train model=finetune --cfg job --resolve

This will print the full configuration file without starting training, useful for debugging and understanding parameters.

Next Steps
==========

- Read the :doc:`user_guide` for detailed usage instructions
- Check the :doc:`training` guide for advanced training options
- Explore the :doc:`api` reference for complete function documentation 