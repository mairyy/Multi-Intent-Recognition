# Multi-Intent-Recognition

<h1>1. Generate relations using Comet</h1>

- Firstly, create environment and install dependencies:

```
conda create --name mir python=3.6
conda activate mir
pip install tensorflow
pip install ftfy==5.1
conda install -c conda-forge spacy
python -m spacy download en
pip install tensorboardX
pip install tqdm
pip install pandas
pip install ipython
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

<h1>2. Running baseline Mag-Bert</h1>

- Install requirements:

```
cd mag-bert
conda create --name mir python=3.8
pip install torch torchvision torchaudio
pip install -r requirements.txt
```

- Run: `sh scripts/run_mag_bert.sh`
