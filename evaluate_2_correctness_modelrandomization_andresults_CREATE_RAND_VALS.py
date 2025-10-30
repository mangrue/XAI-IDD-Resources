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

""" **Setup** """

# Commented out IPython magic to ensure Python compatibility.
#!pip install shap
#!pip install lime

#!pip install evaluate -q
#!pip install datasets -q

from transformers import RobertaConfig, RobertaModel
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
exec(open("gdrive/UTILS/vanillagradients.py").read())

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
eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/jy46604790_fake_dataset_EU_final.csv", token=True, split="train").shuffle(seed=42).select(range(500)) #EU news / EUvsDisinfo

# (3) ENR
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

# (4) EMNAD
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

df = pd.DataFrame(eval_data)
df = df[["text", "label"]]
print(df)

# # # # # # MODEL # # # # # #

"""# **1. Load the dataset, set up the GPU, install the transformer**"""

#path='/content/drive/MyDrive/benchmark/'

# Set random model

# Define paths

#MODEL MUST BE SET WHEN CREATING RESULTS - NOT WHEN CREATING RANDOM VALUES (FROM BASE RoBERTa MODEL)

# # # # # # MODEL # # # # # #
# Instantiate the tokenizer and model
#tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")
#model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
#model_name = 'jy46604790'

#fine-tuned model on EU disinformation
#tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
#model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
#model_name = 'winterForestStump'

#torch.manual_seed(0)
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
#if use_cuda:
#    torch.cuda.manual_seed(0)

print("Using GPU: {}".format(use_cuda))

"""# **Use RoBERTa Architecture with random weights - ONLY NEEDED FOR GENERATING RANDOM DATA**"""

#https://github.com/huggingface/transformers/issues/2649
#model without pretrained weights
#config = RobertaConfig()  #BertConfig()

#See: https://huggingface.co/docs/transformers/en/model_doc/roberta

#RANDOM MODEL
'''
config = RobertaConfig.from_pretrained("roberta-base",  #bert-base-uncased #BertConfig.(...)
                                    num_labels = 2,
                                    output_attentions = False, #defaults to False
                                    output_hidden_states = False, #defaults to False
                                    device_map="auto") #defaults to auto
model_without_Pretrained = RobertaForSequenceClassification(config) #BertConfig()
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")#, max_len=512, truncation=True)
model_without_Pretrained.to(device)
'''

#BASE MODEL (pretrained but not fine-tuned)
model_without_Pretrained = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels = 2) #BertConfig()
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")#, max_len=512, truncation=True)
model_without_Pretrained.to(device)

path='gdrive/benchmark/'

#FOR CREATING RANDOM VALUES - NOT WHEN CREATING RESULTS
#model_name = "RoBERTa"  #random model
model_name = "RoBERTa_base" #base model

#FOR CREATING RESULTS model must be set
#model_name = 'jy46604790'   #ISOT
#model_name = 'winterForestStump'   #OTHERS

#dataset and path_data_created needed for CREATING RANDOM VALUES
#ISOT
'''
dataset='ISOT'
path_data_created='gdrive/benchmark/created_data/ISOT/'
path_csv= path_data_created+model_name+'_ISOT_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_ISOT.sav'
lime_path=path_data_created+model_name+'_lime_values_500_ISOT.sav'
gradient_path=path_data_created+model_name+'_gradient_values_500_ISOT.sav'
outputs_path=path_data_created+model_name+'_original_probs_ISOT_500.pt'
path_labels=path_data_created+model_name+'_labels_ISOT_500.npy'
'''
#EUvsDISINFO
dataset='EUvsDISINFO'
path_data_created='gdrive/benchmark/created_data/EUvsDISINFO/'
path_csv= path_data_created+model_name+'_EUvsDISINFO_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EUvsDISINFO.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EUvsDISINFO.sav'
gradient_path=path_data_created+model_name+'_gradient_values_500_EUvsDISINFO.sav'
outputs_path=path_data_created+model_name+'_original_probs_EUvsDISINFO_500.pt'
path_labels=path_data_created+model_name+'_labels_EUvsDISINFO_500.npy'
'''
#ENR MIX 4 / ENR
dataset='EU_ENR_MIX4'
path_data_created='gdrive/benchmark/created_data/EU_ENR_MIX4/'
path_csv= path_data_created+model_name+'_EU_ENR_MIX4_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EU_ENR_MIX4.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EU_ENR_MIX4.sav'
gradient_path=path_data_created+model_name+'_gradient_values_500_EU_ENR_MIX4.sav'
outputs_path=path_data_created+model_name+'_original_probs_EU_ENR_MIX4_500.pt'
path_labels=path_data_created+model_name+'_labels_EU_ENR_MIX4_500.npy'

#EMNAD MIX 3 / EMNAD
dataset='EU_EMNAD_MIX3'
path_data_created='gdrive/benchmark/created_data/EU_EMNAD_MIX3/'
path_csv= path_data_created+model_name+'_EU_EMNAD_MIX3_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EU_EMNAD_MIX3.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EU_EMNAD_MIX3.sav'
gradient_path=path_data_created+model_name+'_gradient_values_500_EU_EMNAD_MIX3.sav'
outputs_path=path_data_created+model_name+'_original_probs_EU_EMNAD_MIX3_500.pt'
path_labels=path_data_created+model_name+'_labels_EU_EMNAD_MIX3_500.npy'
'''

#STORAGE MANAGEMENT:
#Manually store somewhere - add XAI method name to a directory and move files there

#NEEDED TO BE SET for RESULT GENERATION
#xai_method_name = "SHAP"
#xai_method_name = "LIME"
#xai_method_name = "VANILLA_GRADIENTS"


# # # # # # # # #
#DATA CREATION
# # # # # # # # #

#path_final=path+'final_files/'+dataset+'/'
#path_final=path+'final_files/eval2/'+dataset+'/'+xai_method_name+'/'

######################
#SET STORE PATH
######################
#path_random=path_data_created+'random/'

#RANDOM
#path_random=path_data_created+'random_3/'
#BASE
path_random=path_data_created+'base_1/'

#FOR CREATION OF RANDOM VALUES
df = pd.DataFrame(eval_data)
df = df[["text", "label"]]
print(df)

'''
#ALTERNATIVE VERSION FOR CREATING RESULTS / LOAD but can also use ABOVE code
df = pd.read_csv(path_data_created+'jy46604790_ISOT_500.csv')
labels = np.load(path_data_created+'jy46604790_labels_ISOT_500.npy')
print("TEXT:")
print(df)
print("LABELS:")
print(labels)
'''



"""# **Compute Random SHAP / LIME / InputXGradient values**

**SHAP random**
"""

xai_method_name = "SHAP"
path_final=path+'final_files/eval2/'+dataset+'/'+xai_method_name+'/'

# define a prediction function https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An%20introduction%20to%20explainable%20AI%20with%20Shapley%20values.html#nlp_model
def f(x):
   tv = torch.tensor([tokenizer.encode(v, padding='max_length', max_length=512, truncation=True) for v in x]).cuda()  #use NVIDIA cuda = FASTER compared to only CPU
   outputs = model_without_Pretrained(tv)[0].detach().cpu().numpy()
   scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
   val = sp.special.logit(scores[:,1]) # use one vs rest logit units
   return val
'''
#def predictor(x):
#    outputs = model_without_Pretrained(**tokenizer(x, return_tensors="pt", padding=True, max_length=512, truncation=True))
#    probas = F.softmax(outputs.logits, dim=1).detach().numpy()
#    val = sp.special.logit(probas[:,1])
#    #print(val)
#    return val

#def f_batch(x):
#    val = np.array([])
#    for i in x:
#      val = np.append(val, predictor(i))
#    return val
'''
explainer = shap.Explainer(f, tokenizer)
#explainer = shap.Explainer(f_batch, tokenizer)

#print("LENGTH TOKENS:")
#print(str(len(tokenizer.encode(df['text'][0])[:512])))
#truncation applied

shap_values_rand = explainer(df['text'], batch_size=10)
#shap_values_rand = explainer(({'label': df['label'], 'text': df['text']}), fixed_context=1, batch_size=20)

import os
if not os.path.exists(path_random):
    os.makedirs(path_random)

pickle.dump(shap_values_rand, open((path_random+model_name+"_"+xai_method_name+"_values_rand_"+dataset+"_500.sav"), 'wb'))

print("- - - SHAP random created - - -")

"""**Vanilla Gradients random**"""

#get_gradients(text, model, tokenizer): loaded at the beginning

xai_method_name = "VANILLA_GRADIENTS"
path_final=path+'final_files/eval2/'+dataset+'/'+xai_method_name+'/'

class_names = ['False','True']

def predictor(texts):
  outputs = model_without_Pretrained(**tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda'))
  probas = F.softmax(outputs.logits, dim=1).detach().numpy()
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
gradient_values_rand = []

for text in list(df["text"]):

  text=str(text.encode(encoding="ascii",errors="ignore")) #convert string to ascii to fix encoding issues

  gradients, words, label = get_gradients(text, model_without_Pretrained, tokenizer) #weights, features, label  #use random model

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
  gradient_values_rand.append(exp_inst)

#print(gradient_values_rand)

#Inspect
#str(gradient_values_rand[0])

import os
if not os.path.exists(path_random):
    os.makedirs(path_random)

pickle.dump(gradient_values_rand, open((path_random+model_name+"_"+xai_method_name+"_values_rand_"+dataset+"_500.sav"), 'wb'))

print("- - - VANILLA_GRADIENTS random created - - -")


"""**LIME random**"""


'''
torch.cuda.empty_cache()
model_without_Pretrained = nn.DataParallel(model_without_Pretrained)   #to run on multiple GPUs
'''

model_without_Pretrained.to(device)
print("--- Using GPU: "+str(device)+" ---")

xai_method_name = "LIME"
path_final=path+'final_files/eval2/'+dataset+'/'+xai_method_name+'/'

import numpy as np
import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

from transformers import AutoTokenizer, AutoModelForSequenceClassification

class_names = ['False','True']

def predictor(texts):
  outputs = model_without_Pretrained(**tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to('cuda')) #cuda throws exception when parallelization with GPUs, so outcomment it
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
lime_values_rand = []


for i in range(0, len(eval_data), 1):
  #get explanation
  str_to_predict = eval_data[i]['text']
  explainer = LimeTextExplainer(class_names=class_names)
  exp = explainer.explain_instance(str_to_predict, predictor, num_features=1000, num_samples=40) #1 GPU
  #get weight
  exp_weight = return_weights(exp)
  #get features
  exp_features = return_features(exp)
  #create array
  exp_inst = LimeVals(exp_weight, exp_features)
  lime_values_rand.append(exp_inst)
  print("LAST SAVED COUNT: "+str(i))

pickle.dump(lime_values_rand, open((path_random+model_name+"_"+xai_method_name+"_values_rand_"+dataset+"_500.sav"), 'wb'))

print("- - - LIME random created - - -")

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


#LOAD values and INSPECT
'''
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

#only needed for re-structured usage of LIME/Gradient structure based on SHAP
class SHAPVals(object):
  data = []
  values = []

class SHAPValsRand(object):
  data = []
  values = []

'''


'''
if xai_method_name != "SHAP":
  shap_values_final = SHAPVals()
  for i in shap_values:
    shap_values_final.data.append(i.data)
    shap_values_final.values.append(i.values)
  shap_values = shap_values_final
'''

#Check data
#print(str(len(shap_values.data)))  #len = 1, should be 500; issue with savind re-structured values, gradients are fine BUT LIME not!
#print(len(shap_values.data[80]))
#print(shap_values.data[80])
#print(len(shap_values.values[80]))
#print(shap_values.values[80])
#print(shap_values.data[1][1] + " is the string")
#print(str(shap_values.values[1][1]) + " is the weight")
#print(shap_values.data[1][1])
#print(shap_values.values[1][1])

#Check for any 'nan' values:
#print("NAN values: " + str(any(each!=each for each in shap_values.data)) + str(any(each!=each for each in shap_values.data)))
#FalseFalse = means no NAN values present in the created data

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

# # # # # 
"""# **AFTER CREATING RANDOM VALUES PROCEED PROCEED HERE:**"""

# PROCEED IN OTHER FILE: 'evaluate_2_correctness_modelrandomization_andresults.py'
