# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
#!pip install transformers
#!pip install shap
#!pip install lime

#!pip install evaluate -q
#!pip install datasets -q

#from transformers import BertForSequenceClassification, BertTokenizer
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shap

import torch
import pickle
import pandas as pd
import numpy as np

import scipy as sp
import os
import random

#!cp /content/drive/MyDrive/UTILS/vanillagradients.py .
# %run vanillagradients.py

#from google.colab import drive
#drive.mount('/content/drive')

exec(open("gdrive/UTILS/utils_fake_news.py").read())
exec(open("gdrive/UTILS/vanillagradients.py").read())

#login to hugging face
import os
os.system("huggingface-cli login --token [TOKEN] #--add-to-git-credential")

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, pipeline
from datasets import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from evaluate import evaluator
import evaluate

#libraries for perturbations

#!pip install textattack

from textattack.transformations import WordSwapMaskedLM
from textattack.constraints.pre_transformation import RepeatModification
from textattack.augmentation import Augmenter

print("# # # # # SETUP DONE # # # # #")

path='gdrive/benchmark/'

#SET model name (required)
#model_name = "jy46604790"
model_name = "winterForestStump"
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
'''
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
'''

#NEEDED TO BE SET for RESULT GENERATION
#xai_method_name = "SHAP"
#xai_method_name = "LIME"
#xai_method_name = "VANILLA_GRADIENTS"

if dataset == 'ISOT':
  tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect") #, model_max_length=512) #'roberta-base' = BASE MODEL  #OR: hamzab/roberta-fake-news-classification // both trained on ISOT
  model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
else:
  tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector") #, model_max_length=512)
  model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
#else:
  #tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
  #model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
  #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  #model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

device = torch.device(str("cuda") if torch.cuda.is_available() else "cpu")
model = model.to(device)

path_final=path+'final_files/eval3/'+dataset+'/'
path_perturbed=path_data_created+'perturbed_10percent/'

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

df = pd.read_csv(path_csv)

#eval_data = load_dataset("csv", header=0, data_files="/content/drive/MyDrive/DATASETS/FINAL_DATASETS/"+dataset+"_final_only_title.csv", token=True, split="train").shuffle(seed=42).select(range(10))  #ISOT (only title)
#df = pd.DataFrame(eval_data)
#df = df[["text", "label"]]

#only needed for re-structured usage of LIME/Gradient structure based on SHAP
class LIMEVals(object):
  data = []
  values = []

class GRADIENTVals(object):
  data = []
  values = []

labels = np.load(path_labels)

shap_values = pickle.load(open(shap_path, 'rb'))  #lime_path or gradient_path
lime_values = pickle.load(open(lime_path, 'rb'))
gradient_values = pickle.load(open(gradient_path, 'rb'))

#modifications for LIME values and also GRADIENT values
lime_values_final = LIMEVals()
for i in lime_values:
    lime_values_final.data.append(i.data)
    lime_values_final.values.append(i.values)
gradient_values_final = GRADIENTVals()
for i in gradient_values:
    gradient_values_final.data.append(i.data)
    gradient_values_final.values.append(i.values)

original_probs = torch.load(outputs_path, map_location=torch.device('cpu'))
labels = torch.LongTensor(labels)

import os
if not os.path.exists(path_perturbed):
    os.makedirs(path_perturbed)

"""# **Create perturbations for Stablility (ISOT, EUvsDISINFO, ENR, EMNAD)**#"""

"""**TextAttack**"""

#Use multiple GPUs
#torch.cuda.empty_cache()
#model = nn.DataParallel(model)
#model.to(device)
#print("--- Using GPU: "+str(device)+" ---")

print("Starting TextAttack part...")

#DEFINE AUGMENTER

#transformation = WordSwapMaskedLM(method="clare", tokenizer=tokenizer)

transformation = WordSwapMaskedLM(method="bae", tokenizer=tokenizer) #method="bert-attack", tokenizer=tokenizer)

constraints = [RepeatModification()]

# initiate augmenter
#Parameter details see: https://textattack.readthedocs.io/en/latest/apidoc/textattack.augmentation.html
augmenter = Augmenter(
    transformation=transformation,
    constraints=constraints,
    pct_words_to_swap=0.10, #5% or 20%
    transformations_per_example=100,
    #only sets the transformations per example, doesn't minimize running time
    #use fast augmentation and high yield instead to produce as many augmentations as possible
    high_yield=True,
    fast_augment=True,  #stops additional augmentations when number of transformations per example is reached - DOES NOT IMPROVE SPEED though (SINCE WE run longer text which is the reason why it takes so long), turn high_yield on and then take the number of perturbations needed later (from all generated)
    enable_advanced_metrics=False
)

"""**Load dataset, get random values**"""

#path_final=path+'final_files/eval3/'+dataset+'/'
#path_perturbed=path_data_created+'perturbed/'
df = pd.read_csv(path_csv)

#import os
#if not os.path.exists(path_perturbed):
#    os.makedirs(path_perturbed)

#np.random.seed(42)
# Generate an array of X random integers (size) between two numbers (min and max)
#rand_index = np.random.randint(0, high=500, size=350)  #it was: size=350 #high=1000 (= sample size)
#print(rand_index)

#df['text'][rand_index[0]]

"""**Augment text/TextAttack**"""

#Token limit - truncate text - speeds the whole process up
for i in range(0,len(df['text']),1): #rand_index:
  text = tokenizer.tokenize(df['text'][i]) #tokenizer.encode(df['text'][i], padding=True, truncation=True, max_length=512, return_tensors='pt')
  #print("Batch has a size of: "+str(len(text)))
  #df['text'][i]=text[:512] #truncate to maximum number of 512 tokens
  #print("Truncated batch has a size of: "+str(len(df['text'][i])))
  #transform back to text and truncate to max number of 512 tokens
  #df['text'][i] = tokenizer.convert_tokens_to_string(text[:512])
  df.loc[i, 'text'] = tokenizer.convert_tokens_to_string(text[:512])
  #print("Batch "+str(i)+":"+str(df['text'][i]))

from tqdm import tqdm
#import pickle

from collections import Counter

#To test behavior
#df.loc[0, 'text'] = "This is a text sample that should test what is going on."
#print(str(augmenter.augment(df['text'][0]))) #if high_yield=False use sorted(...)

print("COUNT DF text: "+str(len(df['text'])))
#perp = []
for i in tqdm(range(0,len(df['text']),1)):
  print(str(i) + "  is current index...")
  #print("NEXT: " + str(df['text'][i]) + " /////")
  perturbed_text = augmenter.augment(df['text'][i])
  #print(str(augmenter.augment(df['text'][i])))
  #print(type(perturbed_text))
  #print("Count #1: "+str(len(perturbed_text)))
  #print("Count UNIQUE perturbed texts: "+str(len(Counter(perturbed_text).keys())))
  #print(perturbed_text)  #for testing
  np.save(path_perturbed+dataset+'_'+str(i)+'_augmented.npy', perturbed_text[:100])
  #with open(path_perturbed+dataset+'_'+str(i)+'_augmented.pkl', 'wb') as f:
  #  pickle.dump(perturbed_text, f)
  #perp.append(perturbed_text)
  print(str(i) + " | " + path_perturbed+dataset+'_'+str(i)+'_augmented.npy STORED')

#print("PERTURBED TEXTS SAMPLES:")
#print(str(perp[:5]))

#test
#result=augmenter.augment("This is a sample text to check what is going on with the perturbations.")
#print(str(len(result)))

print("-- PERTURBATIONS DONE --")

#create a list with all perturbed texts and saving it
perturbed_texts = []

original_shaps = []
original_limes = []
original_gradients = []

ind=[]

#path_data='/content/drive/MyDrive/fake-news-adversarial-benchmark/data_created/liar/LIAR'
for i in range(0,len(df['text']),1): #rand_index:
  file_path = path_perturbed + dataset+ '_' + str(i) + '_augmented.npy'
  if os.path.exists(file_path):
    perturbed_texts.append(np.load(file_path))

    original_shaps.append(shap_values[i])   #lime_path or gradient_path
    original_limes.append(lime_values[i])
    original_gradients.append(gradient_values[i])

    ind.append(i)
  else:
    continue

perturbed_texts = np.array(perturbed_texts, dtype="object") #current newer numpy version demands this adaptation to function correct (changed around 2023)
#print(perturbed_texts)

np.save(path_perturbed+dataset+'_all_perturbed.npy', perturbed_texts)

pickle.dump(original_shaps, open(path_perturbed+dataset+'_all_perturbed_original_shap.sav', 'wb'))
pickle.dump(original_limes, open(path_perturbed+dataset+'_all_perturbed_original_lime.sav', 'wb'))
pickle.dump(original_gradients, open(path_perturbed+dataset+'_all_perturbed_original_gradient.sav', 'wb'))

np.save(path_perturbed+dataset+'_all_perturbed_ind.npy', ind)

print("- All code done. -")