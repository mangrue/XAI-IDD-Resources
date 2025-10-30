# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
!pip install shap
!pip install lime

!pip install evaluate -q
#!pip install datasets -q
!pip install -U datasets

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import numpy as np
import scipy as sp
import shap
import pickle
import torch
# Device
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

if torch.cuda.is_available():
  print("CPU IN USE: "+torch.cuda.get_device_name(device))

#NVIDIA A100-SXM4-40GB

from google.colab import drive
drive.mount('/content/drive')
#%cd /content/drive/MyDrive #/Colab_Notebooks

#!cp /content/drive/MyDrive/UTILS/utils_fake_news.py .
#%run utils_fake_news.py

!cp /content/drive/MyDrive/UTILS/vanillagradients.py .
# %run vanillagradients.py

#login to hugging face
!huggingface-cli login --token [TOKEN] #--add-to-git-credential

from datasets import load_dataset
from evaluate import evaluator
import evaluate
from transformers import AutoModelForSequenceClassification, pipeline
from datasets import Dataset
from transformers import AutoTokenizer

#SOURCE: https://github.com/ljyflores/fake-news-adversarial-benchmark/blob/master/utils_fake_news.py

#!cp /content/drive/MyDrive/UTILS/utils_fake_news.py .
#%run utils_fake_news.py

# (1) ISOT
#eval_data = load_dataset("csv", header=0, data_files="/content/drive/MyDrive/DATASETS/FINAL_DATASETS/ISOT_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))                                   #ISOT (title+text)

# (2) EUvsDISINFO data
#eval_data = load_dataset("csv", header=0, data_files="/content/drive/MyDrive/DATASETS/FINAL_DATASETS/jy46604790_fake_dataset_EU_final.csv", token=True, split="train").shuffle(seed=42).select(range(500)) #EU news / EUvsDisinfo

# (3) ENR
#eval_data = load_dataset("csv", header=0, data_files="/content/drive/MyDrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

# (4) EMNAD
eval_data = load_dataset("csv", header=0, data_files="/content/drive/MyDrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

df_data = pd.DataFrame(eval_data)
df_data = df_data[["text", "label"]]
print(df_data)

# # # # # # MODEL # # # # # #
# Instantiate the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")
#model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")

# fine-tuned model on EU disinformation
tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")

#csv_path= path_data_created+'ISOT_1000.csv'
#df_data['text'].to_csv(csv_path)

#df_encode = encode_dataframe(df_data['text'], df_data['label'].tolist())
#print(df_encode)

"""# **Modified Jaccard Similarity**"""

def jacs(init_list1,reit_list2):  #modified: no use of sets, duplicates are counted and the position matters, features the normal jaccard does not fulfill (usually no duplicates and random order due to sets!)
  union = 0
  iter = 0

  for i in range(len(init_list1)):
    union+=1
    if(init_list1[i] == reit_list2[i]):
        iter+=1

  return iter/union

"""#**REITERATION SIMILARITY (& compute Jaccard value in case of differences in sets)**

**SHAP**
"""

# define a prediction function
def predictor(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).cuda()
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

    #outputs = model(**tokenizer(x, return_tensors="pt", padding=True, max_length=512, truncation=True))
    #probas = F.softmax(outputs.logits, dim=1).detach().numpy()
    #val = sp.special.logit(probas[:,1])
    #print(val)
    #return val

#10 rows from ISOT only titles = CPU takes forever (1hour+), with T4 GPU TESLA takes around 1 min.

label = ['False','True'] #[0,1] #{"FAKE": 0, "TRUE": 1}

#REITERATION SIMILARITY
#run same input several times on XAI method - check whether the output is the same

reit_count = 1
shap_values_reit = []
shap_values_init = []

while reit_count < 11:

  print("# # # # # ROUND "+str(reit_count)+" # # # # #")
  explainer = shap.Explainer(predictor, tokenizer)
  model.to(device)
  shap_values_reit == explainer(df_data['text'], batch_size=10)

  if reit_count == 1:  #first iteration save values
    shap_values_init = shap_values_reit
    print("No comparison of SHAP values due to first round ["+str(reit_count)+"]")
  elif shap_values_reit == shap_values_init and reit_count > 1: #reiteration check if equal
    print("Same SHAP values after iteration ["+str(reit_count)+"]")
  else:
    print("Different SHAP values after iteration ["+str(reit_count)+"] - NEW: "+str(shap_values_reit)+" - VS. - INIT: "+str(shap_values_init)) #if not equal show initialization values and new values
    print("Jaccard Similarity:"+str(jacs(shap_values_init, shap_values_reit)))

  reit_count+=1

# Save SHAP values
#pickle.dump(shap_values, open(shap_path, 'wb'))

# Save labels corresponding to SHAP values
#np.save(path_labels, df_data['label'])

"""**LIME**"""

import numpy as np
import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer
import statistics

#from transformers import AutoTokenizer, AutoModelForSequenceClassification

#tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
#model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")

use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
if torch.cuda.is_available():
  print("GPU IN USE: "+torch.cuda.get_device_name(device))
model.to(device)

class_names = ['False','True']

def predictor(texts):
  outputs = model(**tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda'))
  probas = F.softmax(outputs.logits.cpu(), dim=1).detach().numpy()
  return probas

  #outputs = model(**tokenizer(texts, return_tensors="pt", padding=True))
  #probas = F.softmax(outputs.logits, dim=1).detach().numpy()
  #return probas

#GLOBAL Lime explanations - aggregate
#obtain local predictions for a large number of predictions and then average the scores assigned to each feature across all the local explanations to produce a global explanation

import numpy as np
import pandas as pd

def return_weights(exp):
    #get weights from LIME explanation object
    exp_list = exp.as_map()[1]
    #exp_list = sorted(exp_list, key=lambda x: x[0])
    exp_weight = [x[1] for x in exp_list]

    return exp_weight

def return_features(exp):
  #print(exp.as_list())
  exp_list = exp.as_list()
  exp_feature = [x[0] for x in exp_list]

  return exp_feature

#def return_features(exp):
  #get features from LIME explanation object
  #exp_f_list = exp.as_map()[2]

lime_results = pd.DataFrame(columns=['Feature', 'Weight'])

reit_count = 1
lime_values_reit = []
lime_values_init = []

while reit_count < 11:

  weights = []
  features = []

  print("# # # # # ROUND "+str(reit_count)+" # # # # #")

  for i in range(0, len(eval_data), 1):
    #print("Sample no. " + str(i) + " (TEXT: " + eval_data[i]['text'] + ")")
    #get explanation
    str_to_predict = eval_data[i]['text']
    explainer = LimeTextExplainer(class_names=class_names)
    exp = explainer.explain_instance(str_to_predict, predictor, num_features=1000, num_samples=40) #1 GPU with minimum of 24 GB (Google A100 has 40GB)
    #get weight
    exp_weight = return_weights(exp)
    weights.append(exp_weight)
    #get features
    exp_feature = return_features(exp)
    features.append(exp_feature)
    df = pd.DataFrame({'Feature': features[i], 'Weight': weights[i]})
    #print(df)
    lime_results = pd.concat(([lime_results, df]), ignore_index=True)

  #lime_results.insert(0, 'ID', range(1, 1 + len(lime_results)))
  #print(lime_results)

  global_features = lime_results.groupby(["Feature"]).agg({"Weight": "mean"}).reset_index()

  #obtain local predictions for a large number of predictions and then average the scores assigned to each feature across all the local explanations to produce a global explanation
  #mean of the |SHAP| values for each feature
  #aggregation_functions = {'Weight': 'mean', 'Feature': 'first'}
  #global_features = lime_results.groupby(lime_results['ID']).aggregate(aggregation_functions)

  #print(global_features)

  global_features = global_features.sort_values("Weight", ascending=False).reset_index()

  #print(global_features)

  weights = [item for sublist in weights for item in sublist]
  lime_values_reit = np.round(weights,1)

  if reit_count == 1:  #first iteration save values
    lime_values_init = lime_values_reit
    print("No comparison of LIME values due to first round ["+str(reit_count)+"]")
  elif (lime_values_reit == lime_values_init).all() and reit_count > 1: #reiteration check if equal
    print("Same LIME values after iteration ["+str(reit_count)+"]")
  else:
    print("Different LIME values after iteration ["+str(reit_count)+"] - NEW: "+str(lime_values_reit)+" - \nVS. - INIT: "+str(lime_values_init)) #if not equal show initialization values and new values
    print("Jaccard Similarity:"+str(jacs(lime_values_init, lime_values_reit)))

  reit_count+=1

"""**InputXGradient**

###TF Model (do not use - as models based on PyTorch)
"""

#TFRobertaModel or PyTorchModel (preferred since model does not support fully tensorflow)
'''
import tensorflow as tf
from transformers import AutoModel, TFRobertaModel
from transformers import RobertaTokenizer

tokenizer = RobertaTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")
model = TFRobertaModel.from_pretrained("jy46604790/Fake-News-Bert-Detect", from_pt=True)
'''

#get_gradients(text, model, tokenizer): was loaded at the beginning

"""###PT Model (use - as models based on PyTorch)"""

tokenizer = RobertaTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")
model = RobertaForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")

import matplotlib.pyplot as plt

def plot_gradients(tokens, gradients, title):
  """ Plot  explanations
  """
  plt.figure(figsize=(21,3))
  xvals = [ x + str(i) for i,x in enumerate(tokens)]
  colors =  [ (0,0,1, c) for c in (gradients) ]
  # edgecolors = [ "black" if t==0 else (0,0,1, c)  for c,t in zip(gradients, token_types) ]
  # colors =  [  ("r" if t==0 else "b")  for c,t in zip(gradients, token_types) ]
  plt.tick_params(axis='both', which='minor', labelsize=29)
  p = plt.bar(xvals, gradients, color=colors, linewidth=1 )
  plt.title(title)

  #Remove 'Ġ' (explanation: https://github.com/facebookresearch/fairseq/issues/1716 and https://discuss.huggingface.co/t/why-do-i-get-g-when-adding-emojis-to-the-tokenizer/7056)
  char = 'Ġ'
  for idx, ele in enumerate(tokens):
    tokens[idx] = ele.replace(char, '')

  p=plt.xticks(ticks=[i for i in range(len(tokens))], labels=tokens, fontsize=12,rotation=90)

'''
texts = ["The results of the elections appear to favour candidate Atiku",
        "The sky is green and beautiful",
        "The fool doth think he is wise, but the wise man knows himself to be a fool.",
        "Oby ezekwesili was talking about results of the polls in today's briefing",
        "Which party ran the most effective campaign strategy? APC or PDP"]
#texts = sorted(texts, key=len)
'''
texts = list(df_data["text"])

print(texts)

"""###PT Model - Plot results"""

examples = []

for text in texts:

  text=str(text.encode(encoding="ascii",errors="ignore")) #convert string to ascii to fix encoding issues

  gradients, words, label = get_gradients(text, model, tokenizer)

  #Remove 'Ġ' (explanation: https://github.com/facebookresearch/fairseq/issues/1716 and https://discuss.huggingface.co/t/why-do-i-get-g-when-adding-emojis-to-the-tokenizer/7056)
  char = 'Ġ'
  for idx, ele in enumerate(words):
    words[idx] = ele.replace(char, '')

  plot_gradients(words, gradients, f"Prediction: {label.upper()} | {text} ")

  #print(label, text)
  #print(gradients)

  examples.append(
      {"sentence": text,
      "words": words,
       "label": label,
      "gradients": gradients}
  )

"""###PT Model - Consistency evaluation"""

import statistics

examples = []

reit_count = 1
ixg_values_reit = []
ixg_values_init = []

while reit_count < 11:

  print("# # # # # ROUND "+str(reit_count)+" # # # # #")

  for text in texts:

    tokenizer = RobertaTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect") #winterForestStump/Roberta-fake-news-detector
    model = RobertaForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect") #winterForestStump/Roberta-fake-news-detector

    text=str(text.encode(encoding="ascii",errors="ignore")) #convert string to ascii to fix encoding issues

    gradients, words, label = get_gradients(text, model, tokenizer)

    #Remove 'Ġ' (explanation: https://github.com/facebookresearch/fairseq/issues/1716 and https://discuss.huggingface.co/t/why-do-i-get-g-when-adding-emojis-to-the-tokenizer/7056)
    char = 'Ġ'
    for idx, ele in enumerate(words):
      words[idx] = ele.replace(char, '')

    #plot_gradients(words, gradients, f"Prediction: {label.upper()} | {text} ")

    #print(label, text)
    #print(gradients)

    #examples.append(
    #    {"sentence": text,
    #    "words": words,
    #    "label": label,
    #    "gradients": gradients}
    #)

  ixg_values_reit = statistics.mean(gradients) #calculate mean of gradients

  if reit_count == 1:  #first iteration save values
    ixg_values_init = ixg_values_reit
    print("No comparison of IXG values due to first round ["+str(reit_count)+"]")
  elif ixg_values_reit == ixg_values_init and reit_count > 1: #reiteration check if equal
    print("Same IXG values after iteration ["+str(reit_count)+"]")
  else:
    print("Different IXG values after iteration ["+str(reit_count)+"] - NEW: "+str(ixg_values_reit)+" - \nVS. - INIT: "+str(ixg_values_init)) #if not equal show initialization values and new values

  reit_count+=1