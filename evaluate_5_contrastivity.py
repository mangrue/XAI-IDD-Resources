# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
!pip install transformers
!pip install shap
!pip install lime

!pip install evaluate -q
!pip install datasets -q

import pickle
import numpy as np
import pandas as pd
import random
from itertools import chain
import torch
#from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
import itertools
import heapq

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/UTILS/vanillagradients.py .
# %run vanillagradients.py

#login to hugging face
!huggingface-cli login --token [TOKEN] #--add-to-git-credential

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, pipeline
from datasets import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from evaluate import evaluator
import evaluate

"""# **(1) KL DIVERGENCE**"""

# # # # # # MODEL # # # # # #
# Instantiate the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")
#model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
#model_name = 'jy46604790'

#fine-tuned model on EU disinformation
#tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
#model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
model_name = 'winterForestStump'

path='/content/drive/MyDrive/benchmark/'

# Define paths
'''
#ISOT
dataset='ISOT'
path_data_created='/content/drive/MyDrive/benchmark/created_data/ISOT/'
path_csv= path_data_created+model_name+'_ISOT_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_ISOT.sav'
lime_path=path_data_created+model_name+'_lime_values_500_ISOT.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_ISOT.sav'
outputs_path=path_data_created+model_name+'_original_probs_ISOT_500.pt'
path_labels=path_data_created+model_name+'_labels_ISOT_500.npy'
'''
#EUvsDISINFO
dataset='EUvsDISINFO'
path_data_created='/content/drive/MyDrive/benchmark/created_data/EUvsDISINFO/'
path_csv= path_data_created+model_name+'_EUvsDISINFO_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EUvsDISINFO.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EUvsDISINFO.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_EUvsDISINFO.sav'
outputs_path=path_data_created+model_name+'_original_probs_EUvsDISINFO_500.pt'
path_labels=path_data_created+model_name+'_labels_EUvsDISINFO_500.npy'
'''
#ENR MIX 4 / ENR
dataset='EU_ENR_MIX4'
path_data_created='/content/drive/MyDrive/benchmark/created_data/EU_ENR_MIX4/'
path_csv= path_data_created+model_name+'_EU_ENR_MIX4_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EU_ENR_MIX4.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EU_ENR_MIX4.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_EU_ENR_MIX4.sav'
outputs_path=path_data_created+model_name+'_original_probs_EU_ENR_MIX4_500.pt'
path_labels=path_data_created+model_name+'_labels_EU_ENR_MIX4_500.npy'

#EMNAD MIX 3 / EMNAD
dataset='EU_EMNAD_MIX3'
path_data_created='/content/drive/MyDrive/benchmark/created_data/EU_EMNAD_MIX3/'
path_csv= path_data_created+model_name+'_EU_EMNAD_MIX3_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EU_EMNAD_MIX3.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EU_EMNAD_MIX3.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_EU_EMNAD_MIX3.sav'
outputs_path=path_data_created+model_name+'_original_probs_EU_EMNAD_MIX3_500.pt'
path_labels=path_data_created+model_name+'_labels_EU_EMNAD_MIX3_500.npy'
'''

#Run LIME or gradient instead of SHAP:
#shap_path = lime_path
shap_path = gradient_path

#NEEDED TO BE SET
#xai_method_name = "SHAP"
#xai_method_name = "LIME"
xai_method_name = "VANILLA_GRADIENTS"

"""# loading the dataframe, the labels, the shap values and the original probabilities"""

#ONLY needed for LIME/Gradient, NOT for SHAP

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

path_final=path+'final_files/eval5/'+dataset+'/'
df = pd.read_csv(path_csv)
labels = np.load(path_labels)

#DATA SAMPLE text and labels
#eval_data = load_dataset("csv", header=0, data_files="/content/drive/MyDrive/DATASETS/FINAL_DATASETS/"+dataset+"_final_only_title.csv", token=True, split="train").shuffle(seed=42).select(range(10))  #ISOT (only title)
#df = pd.DataFrame(eval_data)
#df = df[["text", "label"]]

#LOAD SHAP values and original probs
shap_values = pickle.load(open(shap_path, 'rb'))
original_probs = torch.load(outputs_path, map_location=torch.device('cpu'))
#print(shap_values[:1])

#load random values
shap_path_random = path_data_created+'base_1/RoBERTa_base_'+xai_method_name+'_values_rand_'+dataset+'_500.sav' #base model with random initialized head for sentiment classification task
#shap_path_random = path_data_created+'random_1/RoBERTa_'+xai_method_name+'_values_rand_'+dataset+'_500.sav' #random 1
#shap_path_random = path_data_created+'random_2/RoBERTa_'+xai_method_name+'_values_rand_'+dataset+'_500.sav' #random 2
#shap_path_random = path_data_created+'random_3/RoBERTa_'+xai_method_name+'_values_rand_'+dataset+'_500.sav' #random 3

shap_values_random = pickle.load(open(shap_path_random, 'rb'))
#print(shap_values_random[:1])

labels=torch.LongTensor(labels)

#ONLY needed for LIME/Gradient, NOT FOR SHAP

#only needed for re-structured usage of LIME/Gradient structure based on SHAP
class SHAPVals(object):
  data = []
  values = []

if xai_method_name != "SHAP":
  shap_values_final = SHAPVals()
  for i in shap_values:
    shap_values_final.data.append(i.data)
    shap_values_final.values.append(i.values)
  shap_values = shap_values_final

class SHAPValsRand(object):
  data = []
  values = []

if xai_method_name != "SHAP":
  shap_values_random_final = SHAPValsRand()
  for i in shap_values_random:
    shap_values_random_final.data.append(i.data)
    shap_values_random_final.values.append(i.values)
  shap_values_random = shap_values_random_final

#shap values
print("Count of values: "+ str(len(shap_values.values)))
print("Sum of non-random values:")
print(shap_values.values[0])
print(sum(shap_values.values[0]))
print("Count of random values: "+ str(len(shap_values_random.values)))
print("Sum of random values:")
print(shap_values_random.values[0])
print(sum(shap_values_random.values[0]))
#red (positive values) corresponds to a more positive review and blue (negative values) a more negative review

#print(len(shap_values.data[5]))
#print(len(shap_values_random.data[5]))

from scipy.special import softmax
import numpy as np

#print(original_probs)

probabilities = softmax(original_probs, axis=1)

#print(probabilities)
#for x,y in probabilities:
#    print("{:.5f}".format(float(x)),"{:.5f}".format(float(y)))

#First one
#print(probabilities[0][0])
#print(probabilities[0][1])
probabilities[0]

"""# **Calculate KL Divergence - Random model --------------------------------**

# **Calculate KL Divergence fine-tuned vs random model when xai method is applied on the data of each**
"""

from matplotlib import pyplot
import numpy as np
from scipy.special import softmax

events = ['FALSE', 'TRUE']

#pos = sum(list(filter(lambda x: x > 0, shap_values.values[0])))
#neg = sum(list(filter(lambda x: x < 0, shap_values.values[0])))
#print(neg)
#print(pos)

#use imported softmax function from scipy special above
'''
def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()
'''

#p_distr = softmax([neg, pos])

print("LEN:")
print(str(len(shap_values.values)))

#print(shap_values.values[0])
#print(shap_values.data[0])

#P = xai method and fine-funed model
list_p=[]
cnt=0
while cnt < len(shap_values.values):
  #print(shap_values.values[cnt])
  p = softmax(shap_values.values[cnt])
  #print(shap_values.values[cnt])
  #print(sum(p))  #should be 1 or almost 1
  list_p.append(p)
  cnt+=1
  #print("Values no. x done:"+str(cnt))

#Q = xai method and random model
list_q=[]
cnt_r=0
while cnt_r < len(shap_values_random.values):
  #print(shap_values_random.values[cnt_r])
  q = softmax(shap_values_random.values[cnt_r])
  #print(shap_values_random.values[cnt_r])
  #print(sum(p))  #should be 1 or almost 1
  list_q.append(q)
  cnt_r+=1
  #print("Values random no. x done:"+str(cnt_r))

print("LEN list_p:")
print(str(len(list_p)))
print("LEN list_q:")
print(str(len(list_q)))

print("Sample (list p):")
print(list_p[:1])
#print(list_p[499])
print("Sample random (list q):")
print(list_q[:1])
#print(list_q[499])

from math import log2

# calculate the kl divergence
def kl_divergence(p, q):
	#len must be equal:
	#print(len(p))
	#print(len(q))
	return sum(p[i] * log2(p[i]/q[i]) for i in range(len(p)))

kl_div_list=[]

i=0
while i < len(list_p):
	# calculate (P || Q)
	kl_pq = kl_divergence(list_p[i], list_q[i])
	#print('KL(P || Q): %.3f bits' % kl_pq)
	kl_div_list.append(kl_pq)
	i+=1
	# calculate (Q || P)
	#kl_qp = kl_divergence(q, p)
	#print('KL(Q || P): %.3f bits' % kl_qp)

print(kl_div_list)
print(len(kl_div_list))

print("Mean KL value:")
print(np.mean(kl_div_list))