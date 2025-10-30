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
shap_path=path_data_created+model_name+'_shap_values_v2_500_ISOT.sav'
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

#df_data['text']
#set sample size above
#csv_path= path_data_created+model_name+'_'+dataset'_500.csv'
#df_data['text'].to_csv(path_csv, index=False)

#df_encode = encode_dataframe(df_data['text'], df_data['label'].tolist())
#print(df_encode)

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
#np.save(path_labels, df_data['label'])