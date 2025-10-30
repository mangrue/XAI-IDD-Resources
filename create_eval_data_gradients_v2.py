# -*- coding: utf-8 -*-

# **Setup Environment**

#!pip install shap
#!pip install lime

#!pip install --upgrade pip

#Install specific libraries
#! pip install torch
#! pip install transformers
#! pip install pycaret
#! pip install pandas
#! pip install numpy
#! pip install pycaret
#! pip install matplotlib
#! pip install -U scikit-learn
#! pip install transformers==2.8.0
#!pip install --upgrade huggingface_hub
#!pip install evaluate -q
#!pip install datasets -q

import pandas as pd
import os
import numpy as np
#Visualisation
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(rc={'figure.figsize':(15,5)})
plt.style.use('fivethirtyeight')
#Feature Selection and Modeling
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
#SHAP/Model Explainations
import shap
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import BertTokenizer, BertForMaskedLM
from transformers.modeling_utils import unwrap_model    #unwrap model from parallelization object (multiple GPUs)
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import torch
import torch.nn as nn

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, pipeline
from datasets import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from evaluate import evaluator
import evaluate

"""# **Create SHAP values for individual datasets (ISOT, EUvsDISINFO, ENR, EMNAD)**

**Setup**
"""

# Commented out IPython magic to ensure Python compatibility.
#!pip install shap
#!pip install lime

#!pip install evaluate -q
#!pip install datasets -q

from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import pandas as pd
import numpy as np
import scipy as sp
import shap
import pickle
import torch
# Device
#use all the available GPUs
device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
  print("GPU IN USE: "+torch.cuda.get_device_name(device))

#!cp /content/drive/MyDrive/UTILS/utils_fake_news.py .
#%run utils_fake_news.py

#!cp /content/drive/MyDrive/UTILS/vanillagradients.py .
# %run vanillagradients.py

exec(open("gdrive/UTILS/utils_fake_news.py").read())
exec(open("gdrive/UTILS/vanillagradients.py").read())  #often results in I/O Error 5, therefore run locally on cluster

#from google.colab import drive
#drive.mount('/content/drive')
#%cd /content/drive/MyDrive #/Colab_Notebooks

#login to hugging face
import os
os.system("huggingface-cli login --token [TOKEN] #--add-to-git-credential")

from datasets import load_dataset
from evaluate import evaluator
import evaluate
from transformers import AutoModelForSequenceClassification, pipeline
from datasets import Dataset
from transformers import AutoTokenizer

print("# # # # # SETUP DONE # # # # #")

###############################################

# Commented out IPython magic to ensure Python compatibility.
#SOURCE: https://github.com/ljyflores/fake-news-adversarial-benchmark/blob/master/utils_fake_news.py

#!cp /content/drive/MyDrive/UTILS/utils_fake_news.py .
# %run utils_fake_news.py

# (1) ISOT
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/ISOT_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))                                   #ISOT (title+text)

# (2) EUvsDISINFO data
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/jy46604790_fake_dataset_EU_final.csv", token=True, split="train").shuffle(seed=42).select(range(500)) #EU news / EUvsDisinfo

# (3) ENR
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

# (4) EMNAD
eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

df_data = pd.DataFrame(eval_data)
df_data = df_data[["text", "label"]]
print(df_data)

# # # # # # MODEL # # # # # #
# Instantiate the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")
#model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
#model_name = 'jy46604790'

#fine-tuned model on EU disinformation
tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
model_name = 'winterForestStump'

path='gdrive/benchmark/'

'''
#ISOT
dataset='ISOT'
path_data_created='gdrive/benchmark/created_data/ISOT/'
path_csv= path_data_created+model_name+'_ISOT_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_ISOT.sav'
lime_path=path_data_created+model_name+'_lime_values_500_ISOT.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_ISOT.sav'
outputs_path=path_data_created+model_name+'_original_probs_ISOT_500.pt'
path_labels=path_data_created+model_name+'_labels_ISOT_500.npy'

#EUvsDISINFO
dataset='EUvsDISINFO'
path_data_created='gdrive/benchmark/created_data/EUvsDISINFO/'
path_csv= path_data_created+model_name+'_EUvsDISINFO_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EUvsDISINFO.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EUvsDISINFO.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_EUvsDISINFO.sav'
outputs_path=path_data_created+model_name+'_original_probs_EUvsDISINFO_500.pt'
path_labels=path_data_created+model_name+'_labels_EUvsDISINFO_500.npy'

#ENR MIX 4 / ENR
dataset='EU_ENR_MIX4'
path_data_created='gdrive/benchmark/created_data/EU_ENR_MIX4/'
path_csv= path_data_created+model_name+'_EU_ENR_MIX4_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EU_ENR_MIX4.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EU_ENR_MIX4.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_EU_ENR_MIX4.sav'
outputs_path=path_data_created+model_name+'_original_probs_EU_ENR_MIX4_500.pt'
path_labels=path_data_created+model_name+'_labels_EU_ENR_MIX4_500.npy'
'''
#EMNAD MIX 3 / EMNAD
dataset='EU_EMNAD_MIX3'
path_data_created='gdrive/benchmark/created_data/EU_EMNAD_MIX3/'
path_csv= path_data_created+model_name+'_EU_EMNAD_MIX3_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EU_EMNAD_MIX3.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EU_EMNAD_MIX3.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_EU_EMNAD_MIX3.sav'
outputs_path=path_data_created+model_name+'_original_probs_EU_EMNAD_MIX3_500.pt'
path_labels=path_data_created+model_name+'_labels_EU_EMNAD_MIX3_500.npy'


#df_data['text']
#set sample size above
#csv_path= path_data_created+model_name+'_'+dataset'_500.csv'
#df_data['text'].to_csv(path_csv, index=False)

#df_encode = encode_dataframe(df_data['text'], df_data['label'].tolist())
#print(df_encode)

"""# **1) Vanilla Gradients (local)**"""

#run model on one GPU (get_gradients() method does not support multiple GPUs)
#model = unwrap_model(model)

#device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
#model.to(device)

#Vanilla Gradients (gradient-based explanations)
#https://medium.com/towards-data-science/basics-gradient-input-as-explanation-bca79bb80de0
#evaluate the contribution of a model component on the model output
#https://edwinwenink.github.io/ai-ethics-tool-landscape/explanations/gradient-based/
#debug a simple model I built to classify text as political or not for a specialized dataset
#e.g. inspect that the model correctly focuses on politics related words/tokens in classifying a text as political
#vanilla gradients or gradient sensitivity
#https://victordibia.com/blog/explain-bert-classification/
#PyTorch’s autograd
#https://medium.com/codex/how-to-compute-gradients-in-tensorflow-and-pytorch-59a585752fb2

#needed also for gradient values to match SHAP data structure
class LimeVals(object):
    instances = []
    total_str = 0  # Class variable to track the total number
    def __init__(self, values, data):
        LimeVals.instances.append(self)
        self.values = values  # Instance variable for the weights                               - values - (weights)
        self.data = data  # Instance variable for the features (tokens)                         - data - (features/tokens)
        LimeVals.total_str += 1  # Increment the total number upon each instantiation
    def __str__(self):
        return f'{self.values} : {self.data}'
    def getArr():
      arr = []
      for i in range(0, len(LimeVals.instances), 1):
        arr_i = LimeVals.instances[i].values,LimeVals.instances[i].data
        arr.append(arr_i)
      return arr

label = ['False','True'] #[0,1]
gradient_values = []

'''
# -*- coding: utf-8 -*-
"""vanillagradients.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1EtRaQS_QBZPmk29Bn1s6u5j3M4Gb1D97
"""

#RobertaForSequenceClassification

import tensorflow as tf
import torch as pt
import numpy as np

#adapted, based on tf code by https://victordibia.com/blog/explain-bert-classification/

def get_gradients(text, model, tokenizer):

  def get_correct_span_mask(correct_index, token_size):
    span_mask = np.zeros((1, token_size))
    span_mask[0, correct_index] = 1
    span_mask = pt.tensor(span_mask, dtype=pt.int32)
    return span_mask

  embedding_matrix = model.to('cuda').roberta.embeddings.word_embeddings.weight.data
  #print(embedding_matrix)
  embedding_matrix = embedding_matrix.cpu().detach().numpy()
  embedding_matrix = pt.from_numpy(embedding_matrix).to('cuda')
  #print(embedding_matrix)
  encoded_tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
  #print(encoded_tokens)
  token_ids = list(encoded_tokens["input_ids"].numpy()[0])
  #print(token_ids)
  vocab_size = embedding_matrix.shape[0] #get_shape()[0]
  #print(vocab_size)

  # convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
  token_ids_tensor = pt.tensor([token_ids], dtype=pt.int32)
  #print(token_ids_tensor)
  token_ids_tensor_one_hot = pt.nn.functional.one_hot(token_ids_tensor.long(), vocab_size)
  #print(token_ids_tensor_one_hot)

  # (i) watch input variable
  token_ids_tensor_one_hot = token_ids_tensor_one_hot.clone().detach().float().requires_grad_(True)

  # multiply input model embedding matrix; allows us do backprop wrt one hot input
  #print(token_ids_tensor_one_hot)
  #print(embedding_matrix)
  inputs_embeds = pt.matmul(token_ids_tensor_one_hot.to(pt.float32).to('cuda'), embedding_matrix)

  #print(inputs_embeds)
  #print(encoded_tokens["attention_mask"])

  # (ii) get prediction
  pred_scores = model(inputs_embeds=inputs_embeds, attention_mask=encoded_tokens["attention_mask"])[0]#.logits
  max_class = pt.argmax(pred_scores, axis=1).cpu().numpy()[0]

  # get mask for predicted score class
  score_mask = get_correct_span_mask(max_class, pred_scores.shape[1])

  # zero out all predictions outside of the correct  prediction class; we want to get gradients wrt to just this class
  predict_correct_class = pt.sum(pred_scores.to('cuda') * score_mask.to('cuda') ) #tf.reduce_sum()
  predict_correct_class.backward()

  # (iii) get gradient of input with respect to prediction class
  grad_res = token_ids_tensor_one_hot.grad
  #print(grad_res)
  gradient_non_normalized = pt.linalg.norm(grad_res, dim=2)
  #print(gradient_non_normalized)

  # (iv) normalize gradient scores and return them as "explanations"
  gradient_tensor = (
      gradient_non_normalized /
      pt.max(gradient_non_normalized) #tf.reduce_max()
  )

  gradients = gradient_tensor[0].numpy().tolist()
  #print(gradients)
  token_words = tokenizer.convert_ids_to_tokens(token_ids)  #text must be in utf-8 or ascii otherwise encoding issues as a result

  prediction_label= "TRUE" if max_class == 1 else "FALSE"
  
  return gradients, token_words , prediction_label
'''

for text in list(df_data["text"]):

  text=str(text.encode(encoding="ascii",errors="ignore")) #convert string to ascii to fix encoding issues
  
  gradients, words, label = get_gradients(text, model, tokenizer) #weights, features, label #loaded at the beginning

  #Remove 'Ġ' (explanation: https://github.com/facebookresearch/fairseq/issues/1716 and https://discuss.huggingface.co/t/why-do-i-get-g-when-adding-emojis-to-the-tokenizer/7056)
  char = 'Ġ'
  for idx, ele in enumerate(words):
    words[idx] = ele.replace(char, '')

  #get weight
  #exp_weight = return_weights(exp)
  #get features
  #exp_features = return_features(exp)
  #create array
  exp_inst = LimeVals(gradients, words)
  gradient_values.append(exp_inst)

#print(gradient_values)
print("DONE with gradients")

#Inspect
#print(str(len(gradient_values[2].data)))
#print(str(len(gradient_values[2].values)))
#print(str(gradient_values[2].data))

#print("Original:")
#print(df_data["text"][2])

# Save Gradient values
pickle.dump(gradient_values, open(gradient_path, 'wb'))

# Save labels corresponding to Gradient values
#np.save(path_labels, df_data['label'])     #same for all XAI methods