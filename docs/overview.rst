.. _overview:

=======
Overview
=======

HotSpot Therapeutics MolE Enhancement
=====================================

This project represents **HotSpot Therapeutics' enhanced implementation** of molecular embeddings, 
building upon and extending the original MolE foundation model architecture. Our development 
focuses on advancing molecular property prediction and drug discovery applications through 
improved model architectures and training methodologies.

**Key Enhancements:**

- Advanced transformer architectures optimized for molecular data
- Enhanced geometric deep learning integration
- Improved pretraining strategies on large molecular datasets  
- Optimized multi-task learning for biological property prediction
- Enhanced inference capabilities for drug discovery workflows

**Original MolE Foundation:**

The underlying MolE architecture combines geometric deep learning with transformers, 
using self-supervised learning on ~842 million molecules followed by multi-task 
training on biological datasets.

Project Summaries
=================

Unsupervised Pretraining
------------------------

.. include:: MOLE_UNSUPERVISED_PRETRAINING_SUMMARY.md
   :parser: myst_parser.sphinx_

Data Pipeline
-------------

.. include:: MOLE_DATA_PIPELINE_SUMMARY.md
   :parser: myst_parser.sphinx_
