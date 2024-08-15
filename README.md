# indoml-bert-baseline

A simple BERT based baseline for DataThon @ [IndoML'24](https://sites.google.com/view/datathon-indoml24).

### Data & Details
Register - [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/19907#learn_the_details), 
get data from [here](https://codalab.lisn.upsaclay.fr/competitions/19907#participate),
download the raw data and store it in a directory (ideally, called `data/`).

### Preprocess
Run
```python
python preprocess.py --data_dir <your_data_directory>
```

### Train & Test
Basic demo, [![colab_logo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karannb/indoml-bert-baseline/blob/main/tutorial.ipynb)

The rest of the code works on all configurations from single CPU, multi-GPU to multi-machine.
```python
python3 trainer.py --output <some_output_column>
```
The code will automatically pick up multiple GPUs.


### All the best!