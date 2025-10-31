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
eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/ISOT_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))                                   #ISOT (title+text)

# (2) EUvsDISINFO data
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/jy46604790_fake_dataset_EU_final.csv", token=True, split="train").shuffle(seed=42).select(range(500)) #EU news / EUvsDisinfo

# (3) ENR
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

# (4) EMNAD
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

df_data = pd.DataFrame(eval_data)
df_data = df_data[["text", "label"]]
print(df_data)

# # # # # # MODEL # # # # # #
# Instantiate the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")
model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
model_name = 'jy46604790'

#fine-tuned model on EU disinformation
#tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
#model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
#model_name = 'winterForestStump'

path='gdrive/benchmark/'


#ISOT
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

#df_data['text']
#set sample size above
#csv_path= path_data_created+model_name+'_'+dataset'_500.csv'
df_data['text'].to_csv(path_csv, index=False)

df_encode = encode_dataframe(df_data['text'], df_data['label'].tolist())
print(df_encode)

'''
"""#  **0) Original results (accuracy)**"""

import itertools

def evaluate(bert_dataloader, bert_model):
    # Generate predictions
    outputs = []
    labels = []
    with torch.no_grad():
        for step, batch in enumerate(bert_dataloader):
            # Unpack batch
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Forward pass
            output = bert_model(b_input_ids, b_input_mask)
            outputs.append(output)

            # Keep labels
            labels.append(b_labels)

    # Stack outputs
    outputs = torch.vstack([item[0].detach() for item in outputs])

    # Stack labels
    labs = [list(i.cpu().numpy()) for i in labels]
    labs = np.array(list(itertools.chain(*labs)))

    return outputs, labs

# Load test dataset into dataloader
batch_size = 10 #32

dataloader = torch.utils.data.DataLoader(df_encode, batch_size=batch_size, shuffle=False)

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")
model.to(device)
# Evaluate
outputs, labs = evaluate(dataloader, model)

torch.save(outputs,outputs_path)

#Check output
outputs, labs

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
#print(str(gradient_values[0]))

# Save Gradient values
pickle.dump(gradient_values, open(gradient_path, 'wb'))

# Save labels corresponding to Gradient values
np.save(path_labels, df_data['label'])     #same for all XAI methods

"""# **2) SHAP (local and global)**"""

#SHAPely values = how each feature impacts the model's prediction
#1) Baseline Value (Prediction): average prediction if none of the features were used, the model’s “neutral” starting point
#2) Marginal Contributions: For each feature, SHAP computes how much the prediction changes when you add that feature to a subset of other features. It does this for all possible subsets, ensuring that every feature gets its fair share of the “credit” for improving the prediction.
#3) Feature SHAP Values: The contribution of each feature is averaged across all these possible combinations. This gives us the SHAP value for that feature — how much it contributed, positively or negatively, to the model’s prediction.
#https://medium.com/biased-algorithms/shap-values-for-random-forest-1150577563c9#:~:text=Baseline%20Value%3A%20First%2C%20SHAP%20calculates,model's%20%E2%80%9Cneutral%E2%80%9D%20starting%20point.

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

#10 rows from ISOT only titles = CPU takes forever (1hour+), with T4 GPU TESLA takes around 1 min., so app. 1.66h for 1000 rows

label = ['False','True'] #[0,1]
explainer = shap.Explainer(predictor, tokenizer)

model.to(device)
shap_values = explainer(df_data['text'], batch_size=10)
print(shap_values)

# Save SHAP values
pickle.dump(shap_values, open(shap_path, 'wb'))

# Save labels corresponding to SHAP values
np.save(path_labels, df_data['label'])

"""**Inspect structure of SHAP values**"""

#import pickle

path='gdrive/benchmark/'

path_final=path+'final_files/'+dataset+'/'
df = pd.read_csv(path_csv)
labels = np.load(path_labels)

#DATA SAMPLE text and labels
#eval_data = load_dataset("csv", header=0, data_files="/content/drive/MyDrive/DATASETS/FINAL_DATASETS/"+dataset+"_final_only_title.csv", token=True, split="train").shuffle(seed=42).select(range(10))  #ISOT (only title)
#df = pd.DataFrame(eval_data)
#df = df[["text", "label"]]

#LOAD SHAP values and original probs
shap_values = pickle.load(open(shap_path, 'rb'))
original_probs= torch.load(outputs_path, map_location=torch.device('cpu'))

labels=torch.LongTensor(labels)

print("DONE with SHAP values")

#Check data
print("\n# # # # # SHAP VALUES # # # # #\n")
print(shap_values[:5])
print("\n# # # # # DATA # # # # #\n")
print(df[:5])
print("\n# # # # # LABELS # # # # #\n")
print(labels[:5])

#Rebuild SHAP data structure

#Data type: object
#.values = array([array([x,  x,  x, x]), array([x,  x,  x, x]), array([x,  x,  x, x]), array([x,  x,  x, x]), array([x,  x,  x, x])], dtype=object) #= 'WEIGHTS'
#.base_values = array([x, x, x, x]) #= BASE VALUES
#.data = (array(['', 'More ', 'Californ', 'ians '], dtype=object)) #= FEATURES

#LIME and InputXGradient have no BASE VALUES, check with metrics 1-3 whether base values needed to compute them or just weihts (values)
#rebuilt this structure in order to also have such data for LIME and InputXGradient in this structured way

#create values out of sample

#transform to dtype object
#class lime_values:
#  values =
#  data =

#and save it then
'''

"""# **3) LIME (local)**"""
#Alternative approach: Two step approach - see details below - due to inconsistant behavior of XAI method LIME

'''
model= nn.DataParallel(model)   #to run on multiple GPUs
model.to(device)
'''

model.to(device)    #run on single GPU

import numpy as np
import lime
import torch
import torch.nn.functional as F
from lime.lime_text import LimeTextExplainer

from transformers import AutoTokenizer, AutoModelForSequenceClassification

#tokenizer = AutoTokenizer.from_pretrained("hamzab/roberta-fake-news-classification")
#model = AutoModelForSequenceClassification.from_pretrained("hamzab/roberta-fake-news-classification")
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

#def return_features(exp):
  #get features from LIME explanation object
  #exp_f_list = exp.as_map()[2]

#already added above
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

#Samples:
#ex1 = LimeVals([1,  2,  3, 4], ['', 'More ', 'Californ', 'ians '])
#ex2 = LimeVals([5,  2,  3, 4], ['', 'Test ', 'Californ', 'ians '])

#print(str(ex1.display_details())+'\n\n')
#for i in range(0, len(LimeVals.instances), 1):
# examples.append(LimeVals.instances[i])
#print(LimeVals.instances[0])

#print(str(LimeVals.getArr(LimeVals)))
#lime_values = LimeVals.getArr()
#print(str(lime_values)+'\n')
#print(lime_values[:2])

label = ['False','True'] #[0,1]
lime_values = []

#CHECK -> "FALSE" -> also by checking there are no NaN values
#print("ANY NaN #### : "+str(df_data['text'].isnull().any().any()))

# Save labels corresponding to LIME values
np.save(path_labels, df_data['label'])     #same for all XAI methods


'''
import os
if not os.path.exists(path_data_created+'PARTS/'):
    os.makedirs(path_data_created+'PARTS/')
'''

for i in range(0, len(eval_data), 1):
  #get explanation
  str_to_predict = eval_data[i]['text']
  #str_to_predict = ' '.join(str_to_predict.split())
  #print("NEXT STRING is: " + str_to_predict)
  explainer = LimeTextExplainer(class_names=class_names)
  #MAX. with 4 NVIDIA RTX5000 (24GB) parallel executed: sample number of 160-220, from 230 on always out of memory (however, still very instable for the 160 samples)
  exp = explainer.explain_instance(str_to_predict, predictor, num_features=1000, num_samples=40) #maximum 40 stable with one 24GB GPU
  #get weight
  exp_weight = return_weights(exp)
  #get features
  exp_features = return_features(exp)
  #create array
  exp_inst = LimeVals(exp_weight, exp_features)
  lime_values.append(exp_inst)
  #PROBLEM:
  #RANDOM ERROR AT RANDOM POSITIONS - IF RERUN SAME STRING IT WORKS: 'ValueError: Input y contains NaN.'
  #Alternative: Save parts and merge later:
  #pickle.dump(lime_values, open(path_data_created+'PARTS/'+model_name+'_lime_values_500_'+dataset+'_COUNT-0-'+str(i)+'.sav', 'wb'))
  print("LAST SAVED COUNT: "+str(i))

#ONLY WORKS WHEN ONE GPU IS USED:
print(lime_values)
print("DONE with LIME values")

#Inspect
#print(str(len(lime_values[0].values)))

# Save LIME values
#pickle.dump(lime_values, open(lime_path, 'wb'))

'''
#Alternative approach (manually combine to total list)

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

lime_values_part1 = pickle.load(open(path_data_created+model_name+'_lime_values_500_'+dataset+'_COUNT-0-499.sav', 'rb'))
#lime_values_part2 = pickle.load(open(path_data_created+model_name+'_lime_values_500_'+dataset+'_COUNT-96-224.sav', 'rb'))
#lime_values_part3 = pickle.load(open(path_data_created+model_name+'_lime_values_500_'+dataset+'_COUNT-225-394.sav', 'rb'))
#lime_values_part4 = pickle.load(open(path_data_created+model_name+'_lime_values_500_'+dataset+'_COUNT-395-491.sav', 'rb'))
#lime_values_part5 = pickle.load(open(path_data_created+model_name+'_lime_values_500_'+dataset+'_COUNT-492-499.sav', 'rb'))

lime_values_total = lime_values_part1 #+lime_values_part2 +lime_values_part3 +lime_values_part4 +lime_values_part5
pickle.dump(lime_values_total, open(lime_path, 'wb'))
print("SAVED TOTAL LIME VALUES")
'''