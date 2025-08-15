Here’s a concise and clear README draft in English based on your description:

---

# README

## Overview

This repository contains scripts for quantum architecture search (QAS) and predictor training. The workflow consists of generating circuits, computing their properties, and training predictors using the proposed methods.

## Workflow

1. **Circuit Sampling and Training**

   * Run `Generate_Data/sampling_training.sh`
     This script performs circuit sampling and trains circuits according to the target task.

2. **Compute Circuit Properties**

   * Run `properties.sh`
     This script calculates the corresponding properties (metrics) for the sampled circuits.

3. **Predictor Training**

   * `Teacher-Student_QAS_TFIM.sh` and `UACS_training_TFIM.sh`
     These scripts train the predictor models for our proposed methods.

4. **Quantum Architecture Search (QAS)**

   * `Teacher-Student_QAS_TFIM.sh` and `UACS_QAS_TFIM.sh`
     These scripts perform the QAS phase to identify optimal circuit architectures.

