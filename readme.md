## Tools:

```bash
pip install sentencepiece
pip install transformers
pip install datasets
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install pandas
pip install numpy
pip install matplotlib
pip install tqdm
pip install beautifulsoup4
pip install requests
pip install wandb
pip install scikit-learn
```

## Prerequisite:
GPU (Otherwise T5 model training will be very slow.)
wandb "account" replace with your account in t5_train.py file.

## Datset pre-processing: 
pre-processing.ipynb

## Scrapping code:
1) scrap.py  (use only wikipedia query)
2) google_scrap.py (Use search + wikipedia query)

## Train:
Run python t5_train.py (To train the model)

## Inference:
t5inference.ipynb (replace the above trained model checkpoint path in code)

## PPT:
NLP_ass1_presentation.pptx contains the overview of approach,results and analysis. 
