# EviSEC: Evidential Spectrum-Aware Contrastive Learning for Out-of-Distribution Detection in Dynamic Graphs

## Abstract
In this study, we explore [Out-of-Distribution Detection in Dynamic Graphs] and analyze it using [Evidential Deep Learning]. We employ the 6 dataset in three download task and implement our methods for a comprehensive analysis of results.

## Dataset
|                     | **# Nodes** | **# Edges** | **# Time Splits** | **Task** |**source**|
|---------------------|-------------|-------------|-------------------|----------|    |
| BC-OTC              | 5,881       | 35,588      | 95 / 14 / 28       | EC       |  |
| BC-Alpha            | 3,777       | 24,173      | 95 / 13 / 28       | EC       |  |
| Reddit              | 55,863      | 858,490     | 122 / 18 / 34      | EC       |  |
| SBM                 | 1,000       | 4,870,863   | 35 / 5 / 10        | LP       |  |
| UCI                 | 1,899       | 59,835      | 62 / 9 / 17        | LP       |  |
| AS                  | 6,474       | 13,895      | 70 / 10 / 20       | LP       |  |
| Elliptic            | 203,769     | 234,355     | 31 / 5 / 13        | NC       |  |
| Brain               | 5,000       | 1,955,488   | 10 / 1 / 1         | NC       |  |

## Code Execution Method

To reproduce this study, the following code execution methods were used:

### 1. Conda Environment Requirements
- Python version: 3.6.13
- Dependencies:


### 2. Data Preprocessing
- The code performs data preprocessing, including data cleaning, normalization, and feature selection.
- To execute data preprocessing, use the following commands:

```python
import pandas as pd
# Example code: Loading and cleaning data
data = pd.read_csv('dataset.csv')
data_cleaned = data.dropna()  # Drop missing values
