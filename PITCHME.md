## EmoContext

### An attention based Deep\
Bi-LSTM network for Emotion Detection

---

## Input Embeddings

* We used pre-trained GloVe model which was trained on 2 Billion tweets. We use this pretrained model to generate word embeddings
for our data set.

* We then feed the word embeddings to the sentence embedding model to get a sentence level representation for each turn of conversation.

---

## Self Attentive Sentence encoder

![Emo1](static/sent_embed.jpeg)

* We use self attention to compute the sentence embedding for each turn of conversation.

---
## Model

![Emo2](static/metwork.jpeg)

* We then concatenate the the three sentence embeddings (one for each
turn of conversation) and feed it to a fully connected neural network 
whose output dimension is equal to the number of output classes.
---

## Result

<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{font-family:Arial, sans-serif;font-size:14px;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg th{font-family:Arial, sans-serif;font-size:14px;font-weight:normal;padding:10px 5px;border-style:solid;border-width:1px;overflow:hidden;word-break:normal;border-color:black;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-uys7{border-color:inherit;text-align:center}
</style>
<table class="tg">
  <tr>
    <th class="tg-c3ow">Approach</th>
    <th class="tg-c3ow" colspan="3">Happy</th>
    <th class="tg-c3ow" colspan="2">Sad</th>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow" colspan="3">Angry</th>
  </tr>
  <tr>
    <td class="tg-baqh"></td>
    <td class="tg-baqh">Precision</td>
    <td class="tg-baqh">Recall</td>
    <td class="tg-baqh">F1</td>
    <td class="tg-baqh">Precision</td>
    <td class="tg-baqh">Recall</td>
    <td class="tg-baqh">F1</td>
    <td class="tg-baqh">Precision</td>
    <td class="tg-baqh">Recall</td>
    <td class="tg-baqh">F1</td>
  </tr>
  <tr>
    <td class="tg-uys7">SS-LSTM</td>
    <td class="tg-uys7">69.51</td>
    <td class="tg-uys7">52.29</td>
    <td class="tg-uys7">59.68</td>
    <td class="tg-c3ow">85.42</td>
    <td class="tg-c3ow">76.63</td>
    <td class="tg-c3ow">80.79</td>
    <td class="tg-c3ow">63.33</td>
    <td class="tg-c3ow">73.55</td>
    <td class="tg-c3ow">71.34</td>
  </tr>
  <tr>
    <td class="tg-uys7"><br>2-BiLSTM<br>+<br>attn</td>
    <td class="tg-uys7">65.05</td>
    <td class="tg-uys7">76.06</td>
    <td class="tg-uys7"><br>70.13</td>
    <td class="tg-c3ow"><br><br>68.09</td>
    <td class="tg-c3ow"><br><br>76.80</td>
    <td class="tg-c3ow"><br><br>72.18</td>
    <td class="tg-c3ow"><br><br>60.56</td>
    <td class="tg-c3ow"><br><br>76.06</td>
    <td class="tg-c3ow"><br><br>71.07</td>
  </tr>
</table>
 ---

## Overall Performance

We compared our work with the SS-LSTM presented in the paper ![Sentiment semantic approach for emotion detection in textual conversations](https://arxiv.org/pdf/1707.06996.pdf). Our network performed at par with SS-LSTM if not better, even though we used only one word embedding matrix i.e. the output from our self attentive neural BiLSTM network. 

The average F1-scores are as follows:

 * SS-LSTM: 71.34
 * Ours = 71.08 


