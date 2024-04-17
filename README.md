# Multi-Intent-Recognition

<h1>1. Generate relations using Comet</h1>

- Firstly, create environment and install dependencies:

```
conda create env --name mir python=3.6
pip install tensorflow
pip install ftfy==5.1
conda install -c conda-forge spacy
python -m spacy download en
pip install tensorboardX
pip install tqdm
pip install pandas
pip install ipython
```

- Activate environment:

```
conda activate mir
```

- Prepare Comet:

```
cd comet-commonsense
bash scripts/setup/get_atomic_data.sh
bash scripts/setup/get_conceptnet_data.sh
bash scripts/setup/get_model_files.sh
python scripts/data/make_atomic_data_loader.py
python scripts/data/make_conceptnet_data_loader.py
```

- Download pre-trained models: https://drive.google.com/open?id=1FccEsYPUHnjzmX-Y5vjCBeyRt1pLo8FB

- Make sure your directory resembles this: https://github.com/mairyy/Multi-Intent-Recognition/blob/main/comet-commonsense/directory.md
    
- Put your text files you want to generate relations into input folder

- Generate relations by running the following code:

```
python generate_relations.py 
```
