
# No prejudice! Fair Federated Graph Neural Networks for Personalized Recommendation

This repository is the implementation of F2PGNN: Fair Federated Graph Neural Networks for Personalized Recommendation.

&#10064; *This paper addresses the pervasive issue of inherent bias within Recommendation Systems (RSs) for different demographic groups without compromising the privacy of sensitive user attributes in Federated Learning (FL) environment with the graph-based model.* 





## Requirements

Our experiments are performed on a machine with AMD EPYC 7282 16-Core Processor @ 2.80GHz with 128GB RAM, 80GB A100 GPU on Linux Server.

To install required packages:

```bash
pip install -r requirements.txt
```
&#10064; *The environment was managed using Anaconda with the following version*
```bash
conda==23.3.1
conda-package-handling==2.0.2
```
&#10064; *The installed tensorflow version automatically detects GPU(s) and if not available it will use CPU instead.*



## Training and Testing

First, change the dataset path according to working directory in the data preprocessing files.

#### Baseline

For the baseline, run the following commands:

```bash
cd Baseline
python F2MF_main.py
```

#### F2PGNN

To run the code of the paper, run the following commands:
```bash
cd F2PGNN_main
python main_table1.py or python main_fig5.py
```




## Results

Each run for different (hyperparameters) settings saves the required files at the given path. 

The estimated running time is from 30 minutes to hours, depending on the dataset size.
