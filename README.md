# indoml-bert-baseline

A simple BERT based baseline for DataThon @ [IndoML'24](https://sites.google.com/view/datathon-indoml24).

### Data & Details
Register - [CodaLab](https://codalab.lisn.upsaclay.fr/competitions/19907#learn_the_details), 
get data from [here](https://codalab.lisn.upsaclay.fr/competitions/19907#participate),
download the raw data and store it in a directory (ideally, called `data/`).

### Preprocess
Run
```python
python src/preprocess.py --data_dir <your_data_directory>
```

### Download BERT model and tokenizer
You also need the BERT model and tokenizer in appropriate directories, run,
```python
python src/downloadBERT.py
```

### Train & Test
Basic demo, [![colab_logo](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/karannb/indoml-bert-baseline/blob/main/tutorial.ipynb)

The rest of the code works on all configurations from single CPU, multi-GPU to multi-machine.
```python
python3 src/trainer.py --output <some_output_column>
```
The code will automatically pick up multiple GPUs, or you can also launch by prefixing it with `CUDA_VISIBLE_DEVICES=x,y,z`.


### All the best!