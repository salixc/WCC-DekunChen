# Baseline Model

This is the baseline model provided by Dr. Jiang Feng ([jeffreyjiang@cuhk.edu.cn](jeffreyjiang@cuhk.edu.cn)).

And there is a [link](https://github.com/SLPcourse/Generated-Text-Detection) to the original task description.

## Brief Introduction

The baseline model is a fine-tuned RoBERTa classification model, which includes RoBERTa transformers and a fully connected layer classifier. Transformers receive embedded input vectors and the semantic information is extracted into the \[CLS\] output vector, also called "Semantic vector", by attention mechanism. The classifier takes the semantic vector as the input. The weight vectors of the predictions of humans and ChatGPT are constantly updated by back-propagation.