# DeBERTa Entity-Focused Fine-Tuning
### SemEval 2017 Task 5, Subtask 2
##### Fine-Grained Sentiment Analysis on Financial News

SOTA model for modeling fine-grained sentiment expressions in financial news articles. CNN-BiLSTM regression head trained on fine-tuned DeBERTa entity embeddings. Refer to _Separating_Representation_from_Classification.pdf_ for more detail. Notebooks include:

* Performance comparison between different base models (BERT, RoBERTa, FinBERT, DeBERTa)
* Experimentation with token pooling strategies (\[CLS\] token vs. target entity token vs. a combination of both)
* NER-based entity masking strategy
* Comparison between attached[^1] vs. detached[^2] regression head architectures

#### Attached:
<img width="210" alt="Screenshot 2023-06-20 at 8 18 45 AM" src="https://github.com/sfuller14/DeBERTa_Entity-Focused_Fine-Tuning/assets/54780092/4bf1587a-99ea-4b9e-a9c4-7580203fdde3">
  
#### Detached:
<img width="435" alt="Screenshot 2023-06-20 at 8 19 49 AM" src="https://github.com/sfuller14/DeBERTa_Entity-Focused_Fine-Tuning/assets/54780092/f42655e8-1f4f-4f4d-982c-d65a4bebdcfd">

#### Experiments & Results:
<img width="787" alt="Screenshot 2023-06-20 at 8 27 04 AM" src="https://github.com/sfuller14/DeBERTa_Entity-Focused_Fine-Tuning/assets/54780092/a00d7078-6c51-4792-98bd-80fb75dbe084">


[^1]: "Attached" classification/regression head -- a single network is used to simultaneously fine-tune DeBERTa and perform classification/regression. The loss from the "classification" phase directly affects "representation" (i.e. the production of fine-tuned final hidden states).

[^2]: "Detached" classification/regression head -- the production of fine-tuned final hidden states is performed using a simple primary network (pooling + dense), then a (completely separate) secondary network is utilized for classification/regression using the output of the primary network as input. 
