# RAMCare

This repository contains the implementation of **RAMCare**, a three-stage framework designed for modeling incomplete multimodal Electronic Health Records (EHRs).

## Repository Structure

- `./RAMCare/`:  
  Contains the core implementation of the RAMCare model, including all modules and training scripts.
- `./analysis/`:  
  Provides the construction pipeline for **aligned condition pairs**, which are used to enhance retrieval and support missing modality recovery in Stage 1.
- `stage_main.py`:  
  The main entry point for training and evaluating the RAMCare framework.

## Main Performance

![performance_all](.\img\performance_all.png)

![performance_missing](.\img\performance_missing.png)

The following tables present the performance of **RAMCare** under four experimental settings, including *training on all dataset / complete subset and testing on all datasets* (Table 1) and *training on all dataset / complete subset and testing on missing subsets* (Table 2).
 These results serve as a supplementary analysis to the main manuscript, with several key observations:

1. Across all settings, RAMCare consistently achieves **state-of-the-art performance**, demonstrating the robustness of our two-stage recovery framework.
2. The results on the missing subsets (as shown in Table 2) indicate that our modality recovery directly enhances the prediction accuracy of incomplete samples.
3. Compared with its performance on the missing subset, RAMCare shows a more substantial gain on the all dataset. This improvement arises because the framework recovers missing-modality samples, alleviating modality imbalance. It not only mitigates the negative impact of incomplete samples on normal complete ones but also indirectly expands the training scale through recovered data, enabling the model to learn more effective fusion representations and thus achieve better overall performance.

## Note

A more detailed description of the codebase, will be released upon the acceptance of the associated paper.
