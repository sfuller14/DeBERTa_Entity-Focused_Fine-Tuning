# Targeted Sentiment Regression on Financial News Articles using DeBERTa + Entity-Focused Fine-Tuning
### SemEval 2017 Task 5, Subtask 2
##### Fine-Grained Sentiment Analysis on Financial News

SOTA model for modeling fine-grained sentiment expressions in financial news articles. Detached CNN-BiLSTM regression head trained on fine-tuned DeBERTa entity embeddings[^1]. Refer to [the PDF](https://github.com/sfuller14/DeBERTa_Entity-Focused_Fine-Tuning/blob/master/Separating_Representation_from_Classification.pdf) for more detail. 

## Notebooks
* [Base model comparison notebook](https://github.com/sfuller14/DeBERTa_Entity-Focused_Fine-Tuning/blob/master/Fine_Grained_Financial_Sentiment_Regression_with_BERT.ipynb)
  * BERT, RoBERTa, FinBERT, DeBERTa
* [Main training/experiments notebook with DeBERTa](https://github.com/sfuller14/DeBERTa_Entity-Focused_Fine-Tuning/blob/master/DeBERTa_Entity-Focused_Fine-Tuning.ipynb):
  * Experimentation with token pooling strategies (\[CLS\] token vs. target entity token vs. a combination of both)
  * Experimentation with NER-based token entity masking strategy
  * Comparison between attached[^2] vs. detached[^3] regression head architectures

## Experiments and Results
The experiments showed that sentiment regression performance was improved by:
* Incorporating into the classification model the final hidden states of both the \[CLS\] token _as well as_ the __masked__ target entity token
* Detaching the classification model from the token-level fine-tuning process
  * In other words, placing complex architectures inside the fine-tuning process _performed worse than_ placing the same complex architecture __after__ the standard (boilerplate ```transformers.BertForSequenceClassification```) pooling + dense layer
  * Intuitively, the error propogation backwards through DeBERTa during training seemed to benefit from a closer/simpler signal, resulting in better inputs for the detached CNN-BiLSTM

The tradeoffs between inference time in production systems and model performance is an interesting area for further research. 

#### Attached Classification/Regression Head example:
<img width="340" alt="Screenshot 2023-06-20 at 8 18 45 AM" src="https://github.com/sfuller14/DeBERTa_Entity-Focused_Fine-Tuning/assets/54780092/4bf1587a-99ea-4b9e-a9c4-7580203fdde3">
  
#### Detached Classification/Regression Head (with entity token replacement) example:
<img width="500" alt="Screenshot 2023-06-20 at 8 19 49 AM" src="https://github.com/sfuller14/DeBERTa_Entity-Focused_Fine-Tuning/assets/54780092/f42655e8-1f4f-4f4d-982c-d65a4bebdcfd">

#### Experiments & Results:
<img width="787" alt="Screenshot 2023-06-20 at 8 27 04 AM" src="https://github.com/sfuller14/DeBERTa_Entity-Focused_Fine-Tuning/assets/54780092/a00d7078-6c51-4792-98bd-80fb75dbe084">


[^1]: For BERT-based models, the final token-level embeddings that are output by the fine-tuned model are referred to as the "final hidden states".

[^2]: "Attached" classification/regression head -- a single network is used to simultaneously fine-tune DeBERTa and perform classification/regression. The loss from the "classification" phase directly affects "representation" (i.e. the production of fine-tuned final hidden states).

[^3]: "Detached" classification/regression head -- the production of fine-tuned final hidden states is performed using a simple primary network (pooling + dense), then a (completely separate) secondary network is utilized for classification/regression using the output of the primary network as input. 
