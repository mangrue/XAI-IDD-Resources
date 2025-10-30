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
eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/ISOT_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))                                   #ISOT (title+text)

# (2) EUvsDISINFO data
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/jy46604790_fake_dataset_EU_final.csv", token=True, split="train").shuffle(seed=42).select(range(500)) #EU news / EUvsDisinfo

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
'''
#https://github.com/huggingface/transformers/issues/2649
#model without pretrained weights
#config = RobertaConfig()  #BertConfig()

config = RobertaConfig.from_pretrained("roberta-base",  #bert-base-uncased #BertConfig.(...)
                                    num_labels = 2,
                                    output_attentions = False,
                                    output_hidden_states = False,
                                    device_map="auto")
model_without_Pretrained = RobertaForSequenceClassification(config) #BertConfig()
tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")#, max_len=512, truncation=True)
model_without_Pretrained.to(device)
'''

path='gdrive/benchmark/'

#FOR CREATING RANDOM VALUES - NOT WHEN CREATING RESULTS
#model_name = "RoBERTa"

#FOR CREATING RESULTS model must be set
model_name = 'jy46604790'   #ISOT
#model_name = 'winterForestStump'   #OTHERS

#dataset and path_data_created needed for CREATING RANDOM VALUES
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

#################################################
#SET RANDOM ROUND
#################################################
#path_random=path_data_created+'random/'
#RANDOM
path_random=path_data_created+'random_'
save_directory = "Random_"
#BASE
#path_random=path_data_created+'base_'
#save_directory = "Base_"


#FOR CREATION OF RANDOM VALUES
df = pd.DataFrame(eval_data)
df = df[["text", "label"]]
print(df)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
xai_methods_names = ["SHAP", "LIME", "VANILLA_GRADIENTS"]

for xai_method_name in xai_methods_names:

    cnt=1
    while cnt <=3 : #BASE: cnt <=1

        path_random=path_data_created+'random_'
        save_directory = "Random_"

        print("############### COUNT ###############")
        print(str(cnt))
        print("############### XAI METHOD NAME ###############")
        print(xai_method_name)
        print("############### BEGIN ###############")

        path_random = path_random+str(cnt)+"/"
        save_directory = save_directory+str(cnt)+"/"

        #LOAD values and INSPECT

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
        """# **AFTER CREATING RANDOM VALUES PROCEED WITH THE FOLLOWING STEPS:**"""

        #Run LIME or gradient instead of SHAP (data input structure is modified as the same):
        if xai_method_name=="VANILLA_GRADIENTS":
            shap_path = gradient_path
            #shap_values_rand=gradient_values_rand  #is done below
        elif xai_method_name=="LIME":
            shap_path = lime_path
            #shap_values_rand=lime_values_rand  #is done below
        #elif xai_method_name=="SHAP"
        #    shap_values_rand=shap_values_rand  #is done below

        #SHAP (or LIME or VANILLA_GRADIENTS)
        shap_values = pickle.load(open(shap_path, 'rb'))
        path_random = path_random+"RoBERTa_"+xai_method_name+"_values_rand_"+dataset+"_500.sav" #path_data_created+'random/shap_values_rand.sav'
        shap_values_rand = pickle.load(open(path_random, 'rb'))

        if xai_method_name != "SHAP":
          shap_values_final = SHAPVals()
          for i in shap_values:
            shap_values_final.data.append(i.data)
            shap_values_final.values.append(i.values)
          shap_values = shap_values_final

        if xai_method_name != "SHAP":
          shap_values_random_final = SHAPValsRand()
          for i in shap_values_rand:
            shap_values_random_final.data.append(i.data)
            shap_values_random_final.values.append(i.values)
          shap_values_rand = shap_values_random_final

        #load random values

        """# **Graphs as overview**"""
        '''
        labels = np.load(path_labels)

        shap_values = pickle.load(open(shap_path, 'rb'))
        original_probs= torch.load(outputs_path, map_location=torch.device('cpu'))

        labels=torch.LongTensor(labels)

        shap.plots.bar(shap_values_rand)

        shap.plots.bar(shap_values_rand)

        shap.plots.bar(shap_values)

        shap.plots.text(shap_values[1])

        shap.plots.text(shap_values_rand[1])
        '''
        """# **Load data - SHAP/LIME/Vanilla Gradients**"""

        #SET model, xai method, and dataset, THEN PROCEED:
        #path_random+model_name+xai_method_name+"_values_rand_"+dataset+"_500.sav"

        path_final = path+'final_files/eval2/'+dataset+'/'

        #check SHAP
        #print("SHAP VALUES - EXAMPLE WEIGHTS: "+str(shap_values[5].values))
        #print("SHAP VALUES RANDOM - EXAMPLE WEIGHTS: "+str(shap_values_rand[5].values))
        #check LIME/GRADIENT
        #print("LIME VALUES - EXAMPLE WEIGHTS: "+str(shap_values.data[5]))
        #print("LIME VALUES RANDOM - EXAMPLE WEIGHTS: "+str(shap_values_rand.data[5]))

        #print("LENGTH: "+str(len(shap_values.values)))
        #print("LENGTH RANDOM: "+str(len(shap_values_rand.values)))

        """# **Get spearman correlation**"""

        #https://github.com/langlrsw/MEED/blob/master/imdb/eval_methods.py [USED]
        #alternative: https://huggingface.co/spaces/evaluate-metric/spearmanr
        def eval_correlation(selection_1, selection_2, flag_abs=False):
            # Spearman Correlation Coefficient
            selection_1, selection_2 = np.reshape(selection_1, -1), np.reshape(selection_2, -1)
            if flag_abs:
                selection_1, selection_2 = np.abs(selection_1), np.abs(selection_2)

            temp_1, temp_2 = selection_1.argsort(), selection_2.argsort()
            ranks_1, ranks_2 = np.empty_like(temp_1), np.empty_like(temp_2)
            ranks_1[temp_1], ranks_2[temp_2] = np.arange(len(selection_1)), np.arange(len(selection_2))

            return np.corrcoef(selection_1, selection_2)[0][1]

        #get global correlation // ABSOLUTE VALUES GLOBAL CORRELATION - spearmanr value and corresponding pvalue
        def flatten(l):
            return [item for sublist in l for item in sublist]

        shap_values_flat=flatten(shap_values.values)
        shap_values_rand_flat=flatten(shap_values_rand.values)

        global_correlation = eval_correlation(shap_values_flat, shap_values_rand_flat, flag_abs=True)
        print("ABSOLUTE VALUES GLOBAL CORRELATION: "+str(global_correlation)+" ("+model_name+" | "+dataset+" | "+xai_method_name+")" + " CNT: " + str(cnt))

        #getting the mean
        #https://stats.stackexchange.com/questions/8019/averaging-correlation-values

        #ABSOLUTE VALUES' MEAN
        mean_correlation_abs = np.mean([eval_correlation(shap_values.values[i], shap_values_rand.values[i], flag_abs=True) for i in range(len(shap_values.values))])
        print("ABSOLUTE VALUES' MEAN GLOBAL CORRELATION: "+str(mean_correlation_abs)+" ("+model_name+" | "+dataset+" | "+xai_method_name+")" + " CNT: " + str(cnt))

        #get global correlation // ORIGINAL SIGN VALUES GLOBAL CORRELATION - spearmanr value
        def flatten(l):
            return [item for sublist in l for item in sublist]

        shap_values_flat=flatten(shap_values.values)
        shap_values_rand_flat=flatten(shap_values_rand.values)

        global_correlation = eval_correlation(shap_values_flat, shap_values_rand_flat, flag_abs=False)
        print("ORIGINAL VALUES GLOBAL CORRELATION: "+str(global_correlation)+" ("+model_name+" | "+dataset+" | "+xai_method_name+")" + " CNT: " + str(cnt))

        #getting the mean
        #https://stats.stackexchange.com/questions/8019/averaging-correlation-values

        #ORIGINAL VALUES' MEAN
        mean_correlation = np.mean([eval_correlation(shap_values.values[i], shap_values_rand.values[i], flag_abs=False) for i in range(len(shap_values.values))])
        print("ORIGINAL VALUES' MEAN GLOBAL CORRELATION: "+str(mean_correlation)+" ("+model_name+" | "+dataset+" | "+xai_method_name+")" + " CNT: " + str(cnt))

        """# **GRAPHS AS AN OVERVIEW**"""   #ONLY SUPPORTS SHAP (only for testing purposes)

        #RANDOM

        if xai_method_name=="SHAP":
            shap.plots.bar(shap_values_rand.abs.mean(0), show = False)
            plt.title((dataset + " | " + xai_method_name + " | " + " Global Top Features Randomization Test (RoBERTa)"), fontsize=14)
            plt.xlabel("mean absolute shap values per feature", fontsize=14)
            plt.rcParams["font.family"] = "Arial"
            plt.gcf().set_size_inches(11.69, 8.27)
            #plt.savefig((path_data_created+dataset+'global_shap_rand.png'), dpi=300, bbox_inches='tight')
            plt.savefig((path_final+xai_method_name+'/'+save_directory+xai_method_name+'_'+"RoBERTa"+"_"+dataset+'_'+'global_rand.png'), dpi=300, bbox_inches='tight')
            #plt.show()


        #Hide FONT warning
        import logging
        logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)
        '''
        shap.plots.bar(shap_values_rand)

        np.random.seed(42)
        rand_index= np.random.randint(0, high=500, size=5) #high=1000
        '''
        #shap.plots.text(shap_values_rand[rand_index[0]]) #OUTPUT WITHOUT GUI = <IPython.core.display.HTML object>

        #NON-RANDOM
        if xai_method_name=="SHAP":
            shap.plots.bar(shap_values.abs.mean(0), show = False)
            plt.title((dataset + " | " + xai_method_name + " | " + " Global Top Features Non-Random (" + model_name + ")"), fontsize=14)
            plt.xlabel("mean absolute shap values per feature", fontsize=14)
            plt.rcParams["font.family"] = "Arial"
            plt.gcf().set_size_inches(11.69, 8.27)
            #plt.savefig((path_data_created+dataset+'global_shap_rand.png'), dpi=300, bbox_inches='tight')
            plt.savefig((path_final+xai_method_name+'/'+save_directory+xai_method_name+'_'+model_name+"_"+dataset+'_'+'global_non-random.png'), dpi=300, bbox_inches='tight')
            #plt.show()

        print("DONE")

        '''
        #Hide FONT warning
        import logging
        logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)

        shap.plots.bar(shap_values)

        np.random.seed(42)
        non_rand_index= np.random.randint(0, high=500, size=5) #high=1000

        shap.plots.text(shap_values[non_rand_index[0]])
        '''

        cnt+=1
