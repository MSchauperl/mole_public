=================================
MolE - Molecular Embeddings
=================================

This project represents **HotSpot Therapeutics' advanced development** building upon the MolE 
foundation model for chemistry. Our work combines geometric deep learning with transformer 
architectures to create enhanced molecular representations. This enhanced implementation 
leverages extensive labeled and unlabeled datasets through improved pretraining strategies, 
first using self-supervised learning on ~842 million molecules, followed by optimized 
multi-task training to better assimilate biological information.

.. image:: https://raw.githubusercontent.com/recursionpharma/mole_public/main/docs/MolE_fig.png
   :alt: MolE Architecture
   :align: center
   :width: 600px

Quick Start
===========

.. code-block:: bash

   # Install MolE
   pip install -e .
   
   # Train a model
   mole_train model=finetune data_file='your_data.csv'
   
   # Make predictions
   from mole import mole_predict
   predictions = mole_predict.predict(smiles=['CCC', 'c1ccccc1'], task='regression')

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   overview
   installation
   quickstart
   user_guide

.. toctree::
   :maxdepth: 2
   :caption: Technical Documentation
   :hidden:

   architecture
   data_pipeline
   training
   pretraining
   metrics

.. toctree::
   :maxdepth: 2
   :caption: Developer Guide
   :hidden:

   developer_guide
   refactoring
   security
   requirements

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api
   modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Resources
   :hidden:

   atlas
   checklist
   troubleshooting

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
