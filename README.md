# EviSEC: Evidential Spectrum-Aware Contrastive Learning for Out-of-Distribution Detection in Dynamic Graphs

## Abstract
In this study, we explore [Out-of-Distribution Detection in Dynamic Graphs] and analyze it using [Evidential Deep Learning]. We employ the 6 datasets in three download task and implement our methods for a comprehensive analysis of results.

## Dataset
6 datasets were used in the paper:
|                     | **# Nodes** | **# Edges** | **# Time Splits** | **Task** |
|---------------------|-------------|-------------|-------------------|----------|
| BC-OTC              | 5,881       | 35,588      | 95 / 14 / 28       | EC       |
| BC-Alpha            | 3,777       | 24,173      | 95 / 13 / 28       | EC       |
| UCI                 | 1,899       | 59,835      | 62 / 9 / 17        | LP       | 
| AS                  | 6,474       | 13,895      | 70 / 10 / 20       | LP       | 
| Elliptic            | 203,769     | 234,355     | 31 / 5 / 13        | NC       | 
| Brain               | 5,000       | 1,955,488   | 10 / 1 / 1         | NC       | 

## Data Source
bitcoin OTC: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-otc.html
bitcoin Alpha: Downloadable from http://snap.stanford.edu/data/soc-sign-bitcoin-alpha.html
uc_irvine: Downloadable from http://konect.uni-koblenz.de/networks/opsahl-ucsocial
autonomous systems: Downloadable from http://snap.stanford.edu/data/as-733.html
elliptic: Please see the instruction to manually prepare the preprocessed version or refer to the following repository that originally proposed the usage of the data: https://arxiv.org/abs/1902.10191
brain: Downloadable from https://www.dropbox.com/sh/33p0gk4etgdjfvz/AACe2INXtp3N0u9xRdszq4vua?dl=0
For downloaded data sets please place them in the 'data' folder.

## Code Execution Method

To reproduce this study, the following code execution methods were used:

### 1. Conda Environment Requirements
- Python version: 3.6.13
- Dependencies:
  ```$ conda create --name <env> --file environment.txt```


### 2. Data Preprocessing
- The code performs data preprocessing, including data cleaning, normalization, and feature selection.
- To execute data preprocessing, use the following commands:

```python
import pandas as pd
# Example code: Loading and cleaning data
data = pd.read_csv('dataset.csv')
data_cleaned = data.dropna()  # Drop missing values
