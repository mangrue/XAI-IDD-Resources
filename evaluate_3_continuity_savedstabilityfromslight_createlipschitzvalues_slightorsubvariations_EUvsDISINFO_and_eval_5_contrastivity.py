# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
#!pip install transformers
#!pip install shap
#!pip install lime

#!pip install evaluate -q
#!pip install datasets -q

#INSTALL THIS TRANSFORMER VERSION (OTHERWISE PAD TO MAX LENGTH NOT WORKING AS IN THIS CODE):
#pip install --upgrade transformers==4.51.3  --break-system-packages

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
'''
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
  tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect") #'roberta-base' = BASE MODEL  #OR: hamzab/roberta-fake-news-classification // both trained on ISOT
  model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
else:
  tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
  model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
#else:
  #tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
  #model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
  #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  #model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")

device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu")
model = model.to(device)

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

#path_perturbation_rate = 'perturbed_5percent/'
path_perturbation_rate = 'perturbed_20percent/'

path_final=path+'final_files/'+dataset+'/'
#path_perturbed=path_data_created+'perturbed/'
path_perturbed=path_data_created+path_perturbation_rate

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


"""# **ATTENTION: Perturbed files/data for evaluation needs to be done to do this part | Prepare/load data**"""

#np.random.seed(42)
#rand_index = np.random.randint(0, high=10, size=20)  #it was: size=350 #high=1000
#rand_index

perturbed_texts = np.load(path_perturbed+dataset+'_all_perturbed.npy', allow_pickle=True)

original_shaps = pickle.load(open(path_perturbed+dataset+'_all_perturbed_original_shap.sav', 'rb'))
original_limes = pickle.load(open(path_perturbed+dataset+'_all_perturbed_original_lime.sav', 'rb'))
original_gradients = pickle.load(open(path_perturbed+dataset+'_all_perturbed_original_gradient.sav', 'rb'))

ind = np.load(path_perturbed+dataset+'_all_perturbed_ind.npy', allow_pickle=True)

"""
original_shaps = []
for i in ind:
  original_shaps.append(shap_values[i])
"""

lipschitz_sample = df.iloc[ind]['text'].copy().tolist()   #limit to 512 tokens as max input for RoBERTa
#lipschitz_sample[:10]

print("OUTPUT of perturbed_texts[0][:10]: ")
print(perturbed_texts[0][:10])
print("OUTPUT of lipschitz_sample[0]: ")
print(lipschitz_sample[0])

print("ORIGINAL SHAPs: ")
print(original_shaps[0])
print("ORIGINAL LIMEs: ")
print(original_limes[0])
print("ORIGINAL GRADIENTs: ")
print(original_gradients[0])

#random.seed(42)
#for sublist in perturbed_texts:
#    random.shuffle(sublist)
#perturbed_texts[0][:10]



"""# **XAI Method: SHAP / LIME / Vanilla Gradients**

**SHAP**
"""

# define a prediction function https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html#nlp_model
def predictor_f(x):
    tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).to(device)
    outputs = model(tv)[0].detach().cpu().numpy()
    scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    val = sp.special.logit(scores[:,1]) # use one vs rest logit units
    return val

explainer_shap = shap.Explainer(predictor_f, tokenizer)

"""**LIME**"""

#prediction function

import numpy as np
import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

from transformers import AutoTokenizer, AutoModelForSequenceClassification

class_names = ['False','True']

def predictor(texts):
  outputs = model(**tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda'))
  probas = F.softmax(outputs.logits.cpu(), dim=1).detach().numpy()
  return probas

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

explainer_lime = LimeTextExplainer(class_names=class_names)
#exp = explainer.explain_instance(str_to_predict, predictor, num_features=1000, num_samples=40) #1 GPU

"""**Vanilla Gradients**"""

#get_gradients(text, model, tokenizer): was loaded at the beginning

"""# from saved TextAttack
based on Naylor et al. https://github.com/mnaylor5/quantifying-explainability

# **DEFINITIONS**
"""

def filter_token(tokens):
  mask = np.logical_and(tokens != 101, tokens != 102)
  mask = np.logical_and(mask, tokens != 0)
  return tokens[mask]

def decode_tokens(tokens):
  string = tokenizer.decode(tokens)
  return string

def filter_decode_tokens(tokens):
  mask = np.logical_and(tokens != 101, tokens != 102)
  mask = np.logical_and(mask, tokens != 0)
  string = tokenizer.decode(tokens[mask])
  return string

def local_lipschitz_torch(original_exp, perturbed_exp, distance):
    # this function is set up to do individual (non-batched) calculations for pytorch tensors
    return torch.norm(original_exp - perturbed_exp).item() / distance

def local_lipschitz(original_exp, perturbed_exp, distance):
    # original_exp (# features) - perturbed_exps (# neighbors, # features) will broadcast across neighbors
    # --- note: this assumes that the perturbed_exp array represents as many documents as input_distances (for the division by distance)
    # taking L2 norm over axis=-1 will return the L2 norm for each perturbed neighbor
    # divide by euclidean distances for each neighbor to get a vector of (# neighbors)
    return np.linalg.norm(original_exp - perturbed_exp, axis=-1) / np.array(distance)

def indices_below_threshold(nested_list, threshold, count_amount=35):
    indices = []
    count=[]
    index=[]
    for i, sublist in enumerate(nested_list):
      count_per_list=0
      indices_per_list = []
      for j, value in enumerate(sublist):
          if value < threshold:
              indices_per_list.append(j)
              count_per_list+=1
      indices.append(indices_per_list)
      count.append(count_per_list)
      if count_per_list > count_amount:
        index.append(i)
    return index, indices

"""# **Check distance to determine EPSILON**"""

d=[]
metrics = []

#ONLY FOR TESTING
#lipschitz_sample = lipschitz_sample[:10] #ONLY FOR TESTING // for testing with smaller sample size
#perturbed_texts = perturbed_texts[:10]

#CHECK
print("LEN LIPSCHITZ SAMPLE and PERTURBED TEXTS:")
print(str(len(lipschitz_sample))+" and "+str(len(perturbed_texts)))

for i in range(0,len(lipschitz_sample)):

  print("--- NOW: sample no. "+str(i)+"---")
  d_i=[]
  original_tokens=tokenizer(lipschitz_sample[i], return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)['input_ids']   #pad to max length necessary due to slight token difference (e.g. 511 vs 512), padding not set to true if pad_to_max_length is true otherwise wouldn't work
  original_text=filter_decode_tokens(original_tokens)
  #print("Original tokens: "+str(original_tokens))
  #print("Original tokens: "+str(len(original_tokens[0])))
  #print("Original text: "+str(len(original_text)))
  #original_embeds = model.bert.embeddings(original_tokens.to(device))
  original_embeds = model.roberta.embeddings(original_tokens.to(device))

  j=0
  #print(len(perturbed_texts[i]))  #output = 100 perturbed samples   #is: 100

  #print("TEXT TO PROCESS:")
  #print(perturbed_texts)

  while j < (len(perturbed_texts[i])):
    perturbed_text = perturbed_texts[i][j]
    #print("PROCESSED TEXT NOW:")
    #print(perturbed_text)
    perturbed_tokens = tokenizer(perturbed_text, return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)['input_ids']   #pad to max length necessary due to slight token difference (e.g. 511 vs 512), padding not set to true if pad_to_max_length is true otherwise wouldn't work

    #print("perturbed_tokens[0]:")
    #print(str(len(perturbed_tokens[0])))   #is: 512
    #print("len(original_tokens[0]):")
    #print(str(len(original_tokens[0])))   #is: 512

    perturbed_embeds = model.roberta.embeddings(perturbed_tokens.to(device))

    d_ij = torch.norm(original_embeds - perturbed_embeds).item() / original_embeds.squeeze().size(0)
    j+=1
    d_i.append(d_ij)

  d.append(d_i)

d = np.array(d, dtype="object") #current newer numpy version demands this adaptation to function correct (changed around 2023)

np.save(path_perturbed+dataset+'_distances.npy', d)

d = np.load(path_perturbed+dataset+'_distances.npy', allow_pickle=True)

print("--- DISTANCES stored ---")

sample_distances_index_1, ind_1 = indices_below_threshold(d, 1)
print(str(len(sample_distances_index_1)))

sample_distances_index_15, ind_15  = indices_below_threshold(d, 1.5)
print(str(len(sample_distances_index_15)))

sample_distances_index_25, ind_25 = indices_below_threshold(d, 0.25, 15)  #0.75 it was
print(str(len(sample_distances_index_25)))

print(str(sample_distances_index_25))

"""# **LOOP - GPU resources needed!**

**SHAP**
"""

cnt_xai_methods = 0   #count successfully processed xai methods

# Commented out IPython magic to ensure Python compatibility.
# %%timeit
# #GPU needed / takes very long / with T4 GPU small sample: 9m39s min. (needs app. 2.5GB from GPU, so not so resource heavy but GPU is needed!)
# 
# #N_DOCS = len(lipschitz_sample)
MIN_NEIGHBORS_PER_DOC = 15 #15 #for testing set to 1 - took 9m39s min., for 15 it likely would take app. 150 min.
EPSILON = 0.25 #0.25 #if it does not work with 0.25 go for 0.75
 
# # # Set store path # SAME FOR ALL XAI METHODS - difference in file names # # # # # # # # # # # # 

store_path = path_perturbed+'threshhold_'+str((EPSILON*100))+'/'
import os
if not os.path.exists(store_path):
  os.makedirs(store_path)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

shap_lipschitz=[]
max_shap_lipschitz_values=[]
successful_transformer_neighbors=0

#sample_distances_index_25 = [0, 1, 2, 4, 8, 9] #small sample to test with lipschitz_sample = lipschitz_sample[:10] #ONLY FOR TESTING // for testing with smaller sample size
#sample_distances_index_25 = [2, 4, 8, 9]

for i in sample_distances_index_25:
  print("--- SHAP No. " + str(i) + " being processed now ---")
  #original_tokens=tokenizer(lipschitz_sample[i], return_tensors='pt')['input_ids']
  original_tokens = tokenizer(lipschitz_sample[i], return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)['input_ids']
  original_text = filter_decode_tokens(original_tokens)
  original_embeds = model.roberta.embeddings(original_tokens.to(device)) #instead of bert
  original_shap = original_shaps[i][:513] #apply limit of model
  successful_transformer_neighbors = 0
  j=0
  while successful_transformer_neighbors < MIN_NEIGHBORS_PER_DOC:
      perturbed_text=perturbed_texts[i][j]

      #print("PERTURBED TEXT:")
      #print(perturbed_text)

      j+=1
      #perturbed_tokens = tokenizer(perturbed_text, return_tensors='pt')['input_ids']
      perturbed_tokens = tokenizer(perturbed_text, return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)['input_ids']

      #print("LEN perturbed tokens")
      #print(str(len(perturbed_tokens[0])))

      #not needed:
      #while len(perturbed_tokens[0]) != len(original_tokens[0]):
      #  perturbed_tokens =  tokenizer(perturbed_texts[i][j], return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)['input_ids']
      #  perturbed_text=perturbed_texts[i][j]
      #  j+=1
      #  #print(j)
      perturbed_embeds = model.roberta.embeddings(perturbed_tokens.to(device))  #instead of bert
      d_ij = torch.norm(original_embeds - perturbed_embeds).item() / original_embeds.squeeze().size(0)
      #print(d_ij)

      #if 0 <= d_ij <= EPSILON: is original but didn't work because if d_ij = 0 there would be a division by 0  #CHECK if not working go for: if 0 < d_ij <= EPSILON:
      if 0 <= d_ij <= EPSILON:
            successful_transformer_neighbors += 1
            perturbed_shap = explainer_shap([perturbed_text])       #different for LIME and Gradient
            #perturbed_shap = explainer_shap(["This is a test whether tokenization works or not."])       #different for LIME and Gradient
            #################################
            #ISSUE - RESOLVED - NOTE: Sentence MUST be in array otherwise problems when running SHAP explanation, even if it is just one sentence!
            #################################

            #print(perturbed_shap)
            # Save SHAP values
            #pickle.dump(perturbed_shap_arr, open(store_path+dataset+'_'+'SHAP'+'_'+'working_file_to_store_shap_vals.npy', 'wb'))
            #perturbed_shap = pickle.load(open(store_path+dataset+'_'+'SHAP'+'_'+'working_file_to_store_shap_vals.npy', 'rb'))

            print(len(original_shap.values))
            print(len(perturbed_shap.values[0]))  #perturbed_shap.values[0]

            #print("ORIGINAL values:")
            #print(original_shap.values)
            #print("PERTURBED values:")
            #print(perturbed_shap.values[0])
            #print("PERTURBED values complete")
            #print(perturbed_shap.values[0])
            #print("All")
            #print(perturbed_shap)

            #E.g. for index 2 difference 164 vs 165, therefore trim the perturbed to the size of the original so it can be broacasted by local_lipschitz function correctly
            #len_orig_vals = len(original_shap.values)
            
            perturbed_shap_arr = perturbed_shap.values[0]
            while len(perturbed_shap_arr) != len(original_shap.values):
              if len(perturbed_shap_arr) > len(original_shap.values):
                print("Info: Perturbed SHAP Vals different len - therefore add neutral weight to original SHAP vals.")
                original_shap.values = np.append(original_shap.values, 0) #add zero / neutral weight at the end
                print("Neutral weight added to original shap vals to match perturbed sample shap vals. -- Updated weights lengths:")
                print(len(original_shap.values))
                print(len(perturbed_shap_arr))
              elif len(perturbed_shap_arr) < len(original_shap.values):
                print("Info: Perturbed SHAP Vals different len (opposite case) - therefore add neutral weight to original SHAP vals.")
                perturbed_shap_arr = np.append(perturbed_shap_arr, 0) #add zero / neutral weight at the end
                print("Neutral weight added to perturbed shap vals to match perturbed sample shap vals. -- Updated weights lengths:")
                print(len(original_shap.values))
                print(len(perturbed_shap_arr))

            #alternative trim but adding zero/neutral weight better so no values are lost
            #shap_lipschitz.append([local_lipschitz(original_shap.values, perturbed_shap.values[0][:len_orig_vals], d_ij)])       #different for LIME and Gradient
            shap_lipschitz.append([local_lipschitz(original_shap.values, perturbed_shap_arr, d_ij)])       #different for LIME and Gradient

            print('successful_transformer_neighbors: '+str(successful_transformer_neighbors))
 
  print(shap_lipschitz)       #different for LIME and Gradient

  np.save((store_path+dataset+'_'+'SHAP'+'_'+str(i)+'_lipschitz.npy'), shap_lipschitz)       #different for LIME and Gradient
 
  max_shap_lipschitz_values.append(max(shap_lipschitz))       #different for LIME and Gradient
  print('max_SHAP_lipschitz_values: '+ str(max_shap_lipschitz_values))
 
  if i == sample_distances_index_25[-1]:
    break

#save max lipschitz values
np.save((store_path+dataset+'_'+'SHAP'+'_max_lipschitz.npy'), max_shap_lipschitz_values)

print("--- SHAP Lipschitz Values stored ---")

cnt_xai_methods += 1

"""**LIME**"""

#N_DOCS = len(lipschitz_sample)
MIN_NEIGHBORS_PER_DOC = 15 #15 #for testing set to 1 - took 9m39s min., for 15 it likely would take app. 150 min.
EPSILON = 0.25 #0.75 #if it does not work with 0.25 go for 0.75

lime_lipschitz=[]
max_lime_lipschitz_values=[]
successful_transformer_neighbors=0

for i in sample_distances_index_25:
  print("--- LIME No. " + str(i) + " being processed now ---")
  #print(lipschitz_sample[i])
  original_tokens=tokenizer(lipschitz_sample[i], return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)['input_ids']
  original_text=filter_decode_tokens(original_tokens)
  original_embeds = model.roberta.embeddings(original_tokens.to(device))
  original_lime = original_limes[i]

  #print("ORIGINAL LIMES LENS")
  #for i in original_limes:
  #  print(str(len(i.values))+" "+str(len(i.data)))

  original_tokens_withoutpaddding=tokenizer(lipschitz_sample[i], return_tensors='pt', truncation=True, max_length=512)['input_ids']
  original_text_toprocess=filter_decode_tokens(original_tokens_withoutpaddding)

  successful_transformer_neighbors = 0
  j=0
  while successful_transformer_neighbors < MIN_NEIGHBORS_PER_DOC:
      perturbed_text=perturbed_texts[i][j]
      #print(perturbed_text)  #text is the same
      j+=1
      perturbed_tokens = tokenizer(perturbed_text, return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)['input_ids']
      #print("LENGTH PERT TOKENS:")
      #print(str(len(perturbed_tokens[0])))  #512 tokens
      perturbed_embeds = model.roberta.embeddings(perturbed_tokens.to(device))
      d_ij = torch.norm(original_embeds - perturbed_embeds).item() / original_embeds.squeeze().size(0)
      #print(d_ij)
            #if 0 <= d_ij
      if 0 < d_ij <= EPSILON:
            successful_transformer_neighbors += 1

            #print("Original Text:")
            #print(original_text_toprocess)
            #print("----------------------")
            #print("Perturbed Text:")
            #print(perturbed_text)

            #LIMITATION TO 512 tokens / LIMITED TEXT LENGTH OF PERTURBATIONS
            #LIME SEEMS TO PROCESS THIS DIFFERENT SO WE GO FOR A FIXED SAMPLE OF THE OVERALL TEXT
            original_lime_exp = explainer_lime.explain_instance(original_text_toprocess, predictor, num_features=1000, num_samples=40)
            perturbed_lime_exp = explainer_lime.explain_instance(perturbed_text, predictor, num_features=1000, num_samples=40) #1 GPU   #507 values for first instance of text in create eval
            
            #get weight
            exp_weight_o = return_weights(original_lime_exp)
            #get features
            exp_features_o = return_features(original_lime_exp)
            #print("Features count: "+str(len(exp_features)))
            #create array
            original_lime_inst = LimeVals(exp_weight_o, exp_features_o)

            #get weight
            exp_weight_p = return_weights(perturbed_lime_exp)
            #get features
            exp_features_p = return_features(perturbed_lime_exp)
            #print("Features count: "+str(len(exp_features)))
            #create array
            perturbed_lime_inst = LimeVals(exp_weight_p, exp_features_p)

            print("Original LIME with full text beyond limit:")
            print(len(original_lime.values))
            print("Original vs perturbed LIME with full text within limit:")
            print(len(original_lime_inst.values))
            print(len(perturbed_lime_inst.values))

            while len(perturbed_lime_inst.values) != len(original_lime_inst.values):
              if len(perturbed_lime_inst.values) > len(original_lime_inst.values):
                print("Info: Perturbed LIME Vals different len - therefore add neutral weight to original LIME vals.")
                original_lime_inst.values = np.append(original_lime_inst.values, 0) #add zero / neutral weight at the end
                print("Neutral weight added to original LIME vals to match perturbed sample LIME vals. -- Updated weights lengths:")
                print(len(original_lime_inst.values))
                print(len(perturbed_lime_inst.values))
              elif len(perturbed_lime_inst.values) < len(original_lime_inst.values):
                print("Info: Perturbed LIME Vals different len (opposite case) - therefore add neutral weight to original LIME vals.")
                perturbed_lime_inst.values = np.append(perturbed_lime_inst.values, 0) #add zero / neutral weight at the end
                print("Neutral weight added to perturbed LIME vals to match perturbed sample LIME vals. -- Updated weights lengths:")
                print(len(original_lime_inst.values))
                print(len(perturbed_lime_inst.values))

            lime_lipschitz.append([local_lipschitz(np.array(original_lime_inst.values), np.array(perturbed_lime_inst.values), d_ij)])
            print('successful_transformer_neighbors: '+str(successful_transformer_neighbors))

  print(lime_lipschitz)
  
  np.save((store_path+dataset+'_'+'LIME'+'_'+str(i)+'_lipschitz.npy'), lime_lipschitz)

  max_lime_lipschitz_values.append(max(lime_lipschitz))
  print('max_LIME_lipschitz_values: '+ str(max_lime_lipschitz_values))

  if i == sample_distances_index_25[-1]:
    break

#save max lipschitz values
np.save((store_path+dataset+'_'+'LIME'+'_max_lipschitz.npy'), max_lime_lipschitz_values)

print("--- LIME Lipschitz Values stored ---")

cnt_xai_methods += 1

"""**Vanilla Gradients**"""

#N_DOCS = len(lipschitz_sample)
MIN_NEIGHBORS_PER_DOC = 15 #15 #for testing set to 1 - took 9m39s min., for 15 it likely would take app. 150 min.
EPSILON = 0.25 #0.75 #if it does not work with 0.25 go for 0.75

gradients_lipschitz=[]
max_gradients_lipschitz_values=[]
successful_transformer_neighbors=0

for i in sample_distances_index_25:
  print("--- Gradients No. " + str(i) + " being processed now ---")
  original_tokens=tokenizer(lipschitz_sample[i], return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)['input_ids']
  original_text=filter_decode_tokens(original_tokens)
  original_embeds = model.roberta.embeddings(original_tokens.to(device))
  original_gradient = original_gradients[i]
  successful_transformer_neighbors = 0
  j=0
  while successful_transformer_neighbors < MIN_NEIGHBORS_PER_DOC:
      perturbed_text=perturbed_texts[i][j]
      j+=1
      perturbed_tokens = tokenizer(perturbed_text, return_tensors='pt', truncation=True, max_length=512, pad_to_max_length=True)['input_ids']
      perturbed_embeds = model.roberta.embeddings(perturbed_tokens.to(device))
      d_ij = torch.norm(original_embeds - perturbed_embeds).item() / original_embeds.squeeze().size(0)
      #print(d_ij)
            #if 0 <= d_ij
      if 0 < d_ij <= EPSILON:
            successful_transformer_neighbors += 1
            perturbed_text=str(perturbed_text.encode(encoding="ascii",errors="ignore")) #convert string to ascii to fix encoding issues
            gradients, words, label = get_gradients(perturbed_text, model, tokenizer)
            #Remove 'Ġ' (explanation: https://github.com/facebookresearch/fairseq/issues/1716 and https://discuss.huggingface.co/t/why-do-i-get-g-when-adding-emojis-to-the-tokenizer/7056)
            char = 'Ġ'
            for idx, ele in enumerate(words):
                words[idx] = ele.replace(char, '')
            #create array
            perturbed_gradients_inst = LimeVals(gradients, words)
            
            print(len(original_gradient.values))
            print(len(perturbed_gradients_inst.values))
            
            while len(perturbed_gradients_inst.values) != len(original_gradient.values):
              if len(perturbed_gradients_inst.values) > len(original_gradient.values):
                print("Info: Perturbed Gradient Vals different len - therefore add neutral weight to original Gradient vals.")
                original_gradient.values = np.append(original_gradient.values, 0) #add zero / neutral weight at the end
                print("Neutral weight added to original Gradient vals to match perturbed sample Gradient vals. -- Updated weights lengths:")
                print(len(original_gradient.values))
                print(len(perturbed_gradients_inst.values))
              elif len(perturbed_gradients_inst.values) < len(original_gradient.values):
                print("Info: Perturbed Gradient Vals different len (opposite case) - therefore add neutral weight to original Gradient vals.")
                perturbed_gradients_inst.values = np.append(perturbed_gradients_inst.values, 0) #add zero / neutral weight at the end
                print("Neutral weight added to perturbed Gradient vals to match perturbed sample Gradient vals. -- Updated weights lengths:")
                print(len(original_gradient.values))
                print(len(perturbed_gradients_inst.values))

            gradients_lipschitz.append([local_lipschitz(np.array(original_gradient.values), np.array(perturbed_gradients_inst.values), d_ij)])
            print('successful_transformer_neighbors: '+str(successful_transformer_neighbors))

  print(gradients_lipschitz)

  np.save((store_path+dataset+'_'+'Gradients'+'_'+str(i)+'_lipschitz.npy'), gradients_lipschitz)

  max_gradients_lipschitz_values.append(max(gradients_lipschitz))
  print('max_Gradients_lipschitz_values: '+ str(max_gradients_lipschitz_values))

  if i == sample_distances_index_25[-1]:
    break

#save max lipschitz values
np.save((store_path+dataset+'_'+'Gradients'+'_max_lipschitz.npy'), max_gradients_lipschitz_values)

print("--- GRADIENT Lipschitz Values stored ---")

cnt_xai_methods += 1

if cnt_xai_methods == 3:
  print("--- All XAI methods successfully processed. Lipschitz Values stored. ---")
