# Sentiment-Analysis-Using-CT-BERT-and-Auxiliary-Sentence-Approach
Code for "Sentiment Analysis on Covid Tweets Using COVID-Twitter-BERT with Auxiliary Sentence Approach", accepted at ACMSE 2021.

## Install
```bash
git clone https://github.com/bubblemans/Sentiment-Analysis-Using-CT-BERT-and-Auxiliary-Sentence-Approach.git
```

### Install Depencencies
```bash
virtualenv env
source env/bin/activate
pip install -r requirements.txt

export CUDA_HOME=/usr/local/cuda-10.1
rm apex
git clone https://github.com/NVIDIA/apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./apex
```


## Usage
```bash
python single_classifier.py
python pair_classifier.py
```