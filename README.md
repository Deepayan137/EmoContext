## EmoContext
Course project for IRE 2018


##3 Requirements

* python 3.5+
* pytorch 0.4
* tqdm

Download pretrained Glove Word Embeddings from http://nlp.stanford.edu/data/glove.twitter.27B.zip

```
pip install -r requirements

```

### Usage

```
python3 -m train_sep_turns.py 

```

### Parameters

```
--inp=25(default) word embedding size
--lr = 1e-3
--inp = 25
--hidden = 256
--out = 4
--depth = 2
--epochs = 20
--filters = 100

```

### Model

![Sent_embeding](sent_embed.jpeg)

Self attentive neural net for Sentenc embedding

![Network](static/network.jpeg)

### For more info please visit ![Project page](https://deepayan137.github.io/category/projects.html)