import numpy as np
import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer, logging
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import colormaps
import matplotlib.colors as colors


def get_prediction(model, input_ids, attention_mask):
    # Get the predicted label using the input_ids and attention_mask
    outputs = model(input_ids, attention_mask=attention_mask)
    predicted_label = np.argmax(outputs.logits.detach().cpu().numpy())

    # Fully connected layer classification
    fc_layer = model.classifier
    W = fc_layer.out_proj.weight.detach().cpu().numpy()

    # CLS vector (Output of RoBERTa model)
    outputs2 = model.roberta(input_ids, attention_mask)
    semantic_vector = outputs2.last_hidden_state.mean(dim=1).squeeze().detach().cpu().numpy()

    return W, semantic_vector, predicted_label


# Concatenate the tokens to words
def wordConcatenate(tokens, V):
    startCh = "Ġ"  # special mark for seperation in baseline model
    tokens[0] = startCh + tokens[0]
    words = []; vectors = []
    now = 0
    while True:
        if now >= len(tokens): break
        tail = now
        if tokens[now][0] == startCh:
            while tail < len(tokens) - 1:
                if tokens[tail+1][0].isalpha() and tokens[tail+1][0] != startCh:
                    tail += 1
                else: break
        
        # Concatenate a word and use the average of tokens vectors
        words.append("".join(tokens[now: tail+1]).replace(startCh, ""))
        vectors.append(list(np.mean(V[now: tail+1], axis=0)))
        now = tail+1
    return words, vectors


def getContribution(test_sentence, tokenizer, model, device):
    # Preprocess the text sentence
    inputs = tokenizer.encode_plus(test_sentence, add_special_tokens=True, return_tensors="pt", max_length=512, truncation=True)
    input_ids, attention_mask = inputs["input_ids"].to(device), inputs["attention_mask"].to(device)

    # Tokens and embedded vectors
    tokens = tokenizer.convert_ids_to_tokens(input_ids.numpy().reshape(-1,)[1:-1])
    V = model.get_input_embeddings()(input_ids).squeeze(0).detach().cpu().numpy()[1:-1]
    tokens, V = wordConcatenate(tokens, V)
    # tokens = [s.replace("Ġ", "") for s in tokens]  # remove special remarks

    W, S, predicted_label = get_prediction(model, input_ids, attention_mask)
    WS = np.zeros((2, 768))
    WS[0] = W[0]*S; WS[1] = W[1]*S  # 0 for humans, 1 for ChatGPT

    # Calculate the cosine similarity with each word
    CS = [cosine_similarity(np.array(v).reshape(1, -1), WS[predicted_label].reshape(1, -1)) for v in V]
    CS = np.array(CS).reshape(-1, )

    # Min-Max scale
    CS = (CS - np.min(CS)) / (np.max(CS) - np.min(CS))

    return predicted_label, tokens, CS


# Plot the text
def getHtml(predicted_label, tokens, CS):
    # Use different style of color to represent human / ChatGPT
    colorRange = ["Blues", "Oranges"]
    textColor = ["#FF6600", "#3366FF"]
    map_vir = colormaps.get_cmap(colorRange[predicted_label])

    # Get the color of each word according to the contribution
    hex_colors = [colors.rgb2hex(color) for color in map_vir(CS)]

    # Form the html for display
    html = ""
    for i in range(len(tokens)):
        html += f'<span style="background-color:{hex_colors[i]}; color: {textColor[predicted_label]}; font-size: 25px; font-weight: bold; padding:0.01px;">{" "+tokens[i]+" "}</span> '
    return html


# Plot the graph in detail
def plotDetail(predicted_label, tokens, CS):
    # Use different color to represent human / ChatGPT
    colorRange = ["Blues", "Oranges"]
    map_vir = colormaps.get_cmap(colorRange[predicted_label])
    CS = list(CS)

    # 20 words for each line in the graph
    if len(tokens) % 20 > 0:
        for i in range(20 - len(tokens) % 20):
            tokens.append("")
            CS.append(0)
    numFig = int(len(tokens) / 20)

    # Only one line of graph
    if numFig == 1:
        plt.figure(figsize=(18, 5))
        plt.bar(range(20), CS, color=map_vir(CS))
        plt.xticks(ticks=range(20), labels=tokens, rotation=20)
    
    # More than one line of graph
    else:
        fig, ax = plt.subplots(numFig, 1, figsize=(18, 5 * numFig))
        for i in range(numFig):
            left = 20 * i
            right = 20 * (i+1)
            ax[i].set_ylim(0, 1)
            ax[i].bar(range(20), CS[left: right], color=map_vir(CS[left: right]))
            ax[i].set_xticks(ticks=range(20), labels=tokens[left: right], rotation=20)