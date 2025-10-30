# -*- coding: utf-8 -*-

# Commented out IPython magic to ensure Python compatibility.
#!pip install transformers
#!pip install shap
#!pip install lime

#!pip install evaluate -q
#!pip install datasets -q

#from transformers import BertForSequenceClassification, BertTokenizer
'''
from transformers import RobertaForSequenceClassification, RobertaTokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import shap

import torch
import pickle

'''

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

'''
exec(open("gdrive/UTILS/utils_fake_news.py").read())
exec(open("gdrive/UTILS/vanillagradients.py").read())

#login to hugging face
import os
os.system("huggingface-cli login --token [TOKEN] #--add-to-git-credential")
'''

'''
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
'''

print("# # # # # SETUP DONE # # # # #")

path='gdrive/benchmark/'

'''
#SET model name (required)
model_name = "jy46604790"
#model_name = "winterForestStump"

#ISOT
dataset='ISOT'
path_data_created='gdrive/benchmark/created_data/ISOT/'
path_csv= path_data_created+model_name+'_ISOT_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_ISOT.sav'
lime_path=path_data_created+model_name+'_lime_values_500_ISOT.sav'
gradient_path=path_data_created+model_name+'_gradient_values_500_ISOT.sav'
outputs_path=path_data_created+model_name+'_original_probs_ISOT_500.pt'
path_labels=path_data_created+model_name+'_labels_ISOT_500.npy'


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


#NEEDED TO BE SET for RESULT GENERATION
#xai_method_name = "SHAP"
#xai_method_name = "LIME"
#xai_method_name = "VANILLA_GRADIENTS"
'''

'''
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

path_final=path+'final_files/'+dataset+'/'
#path_perturbed=path_data_created+'perturbed/'
path_perturbed=path_data_created+'perturbed_5percent/'

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
'''

'''
import os
if not os.path.exists(path_perturbed):
    os.makedirs(path_perturbed)
'''

##############################################################
# SLIGHT PERTURBATIONS VS SUBSTANTIAL PERTURBATIONS - MODIFY #
##############################################################
#path_perturbation_rate = 'perturbed_5percent/'
path_perturbation_rate = 'perturbed_20percent/'

#path_final=path+'final_files/'+dataset+'/'

#path_perturbed=path_data_created+'perturbed/'
#path_perturbed=path_data_created+path_perturbation_rate

"""# **Lipschitz Results**

# **lipschitz read**
"""

#xai_method = "SHAP"
#xai_method = "LIME"
#xai_method = "Gradients"

import os
import re

import matplotlib.pyplot as plt

store_path = 'gdrive/benchmark/'

xai_methods = ["SHAP", "LIME", "Gradients", "SHAP"] #matplot doesn't update font size correctly at first run (known issue) for plt.rcParams['ytick.labelsize'] = 20; therefore run SHAP twice; then correct

for xai_method in xai_methods:

  print("##### XAI METHOD TO BE PROCESSED: "+xai_method+" #####")

  #lip_ISOT = np.load('/content/drive/MyDrive/benchmark/created_data/ISOT/perturbed/threshhold_25.0/ISOT'+'_'+xai_method+'_3_lipschitz.npy')  #ISOT_53_lipschitz.npy #change threshold to 75 if it does not work with 25
  #lip_EUvsDISINFO = np.load('/content/drive/MyDrive/benchmark/created_data/EUvsDISINFO/perturbed/threshhold_25.0/EUvsDISINFO'+'_'+xai_method+'_119_lipschitz.npy')
  #lip_EU_ENR_MIX4 = np.load('/content/drive/MyDrive/benchmark/created_data/EUvsDISINFO/perturbed/threshhold_25.0/EU_ENR_MIX4'+'_'+xai_method+'_119_lipschitz.npy')
  #lip_EU_EMNAD_MIX3 = np.load('/content/drive/MyDrive/benchmark/created_data/EUvsDISINFO/perturbed/threshhold_25.0/EU_EMNAD_MIX3'+'_'+xai_method+'_119_lipschitz.npy')

  list_of_files_ISOT = os.listdir('gdrive/benchmark/created_data/ISOT/'+path_perturbation_rate+'threshhold_25.0/')
  list_of_files_EUvsDISINFO = os.listdir('gdrive/benchmark/created_data/EUvsDISINFO/'+path_perturbation_rate+'threshhold_25.0/')
  list_of_files_EU_ENR_MIX4 = os.listdir('gdrive/benchmark/created_data/EU_ENR_MIX4/'+path_perturbation_rate+'threshhold_25.0/')
  list_of_files_EU_EMNAD_MIX3 = os.listdir('gdrive/benchmark/created_data/EU_EMNAD_MIX3/'+path_perturbation_rate+'threshhold_25.0/')

  #print(list_of_files_ISOT[:10])

  print("-- Reading file names of all directories completed. --")

  #get last saved file with largest number as final one to read for getting all lipschitz values
  def extract_max_number(files,dataset):
    #s = re.findall(r"^"+dataset+"_"+xai_method+"_"+".(\\d+)",f)
    #n = [int(x.split(':')[1]) for x in l]
    max_num = []

    for filename in files:
        #print("Filename: "+filename)

        #searchstr = r"/"+dataset+"_"+xai_method+"\\_"+"(\\d+)"  #fails for enr string patterns
        searchstr = r"("+dataset+"\\w)"+xai_method+"\\w(\\d+)"  #works

        #print("Searchstring is: "+searchstr)
        num_s = re.search(searchstr,filename)
        #print("Regex output: "+str(num_s))

        if num_s != None:
          #print("Regex output group 0: "+str(num_s.group(0)))
          #print("Regex output group 1: "+str(num_s.group(1)))
          #print("Regex output group 2: "+str(num_s.group(2)))
          num = int(num_s.group(2))
          #print("Value of num:"+str(num))
          max_num.append(num)
          #print("Liste max_num: "+str(max_num))

    #print(str(max_num))
    return max(max_num)

  print("-- OVERVIEW MAX PER DATASET AND METHOD: --")

  max_n_ISOT = extract_max_number(list_of_files_ISOT,"ISOT")
  print("MAX ISOT | "+xai_method+" is: "+str(max_n_ISOT))
  max_n_EUvsDISINFO = extract_max_number(list_of_files_EUvsDISINFO,"EUvsDISINFO")
  print("MAX EUvsDISINFO | "+xai_method+" is: "+str(max_n_ISOT))
  max_n_EU_ENR_MIX4 = extract_max_number(list_of_files_EU_ENR_MIX4,"EU_ENR_MIX4")
  print("MAX EU_ENR_MIX4 | "+xai_method+" is: "+str(max_n_ISOT))
  max_n_EU_EMNAD_MIX3 = extract_max_number(list_of_files_EU_EMNAD_MIX3,"EU_EMNAD_MIX3")
  print("MAX EU_EMNAD_MIX3 | "+xai_method+" is: "+str(max_n_ISOT))

  print("-- Extracting max number of files in directories completed. --")

  lip_ISOT = np.load('gdrive/benchmark/created_data/ISOT/'+path_perturbation_rate+'threshhold_25.0/ISOT'+'_'+xai_method+'_'+str(max_n_ISOT)+'_lipschitz.npy')  #ISOT_53_lipschitz.npy #change threshold to 75 if it does not work with 25
  lip_EUvsDISINFO = np.load('gdrive/benchmark/created_data/EUvsDISINFO/'+path_perturbation_rate+'threshhold_25.0/EUvsDISINFO'+'_'+xai_method+'_'+str(max_n_EUvsDISINFO)+'_lipschitz.npy')
  lip_EU_ENR_MIX4 = np.load('gdrive/benchmark/created_data/EU_ENR_MIX4/'+path_perturbation_rate+'threshhold_25.0/EU_ENR_MIX4'+'_'+xai_method+'_'+str(max_n_EU_ENR_MIX4)+'_lipschitz.npy')
  lip_EU_EMNAD_MIX3 = np.load('gdrive/benchmark/created_data/EU_EMNAD_MIX3/'+path_perturbation_rate+'threshhold_25.0/EU_EMNAD_MIX3'+'_'+xai_method+'_'+str(max_n_EU_EMNAD_MIX3)+'_lipschitz.npy')

  print("-- Loading final lipschitz files completed. Proceeding with getting maximum lipschitz numbers per dataset. --")

  '''
  #ONLY USE THIS FOR GETTING THE NEXT MAX LOCAL LIPSCHITZ VALUES AND DON'T APPLY BELOW CODE WHEN BUILDING THE BOXPLOTS
  #ONLY AFFECTS SHAP - normal behavior as can be seen in other studies SHAP has an issue here
  if xai_method=="SHAP":
    #np.set_printoptions(threshold=np.inf)
    #print(lip_ISOT)
    lip_ISOT = np.where(np.isfinite(lip_ISOT), lip_ISOT, 0)
    #print("ADJUSTED lip_ISOT replace nan with value zero (0):")
    #print(lip_ISOT)
    lip_EUvsDISINFO = np.where(np.isfinite(lip_EUvsDISINFO), lip_EUvsDISINFO, 0)
    lip_EU_ENR_MIX4 = np.where(np.isfinite(lip_EU_ENR_MIX4), lip_EU_ENR_MIX4, 0)
    lip_EU_EMNAD_MIX3 = np.where(np.isfinite(lip_EU_EMNAD_MIX3), lip_EU_EMNAD_MIX3, 0)
  '''

  #max lipschitz values
  print("MAX ISOT | "+xai_method+":")
  print(str(np.max(lip_ISOT)))
  print("MAX EUvsDISINFO | "+xai_method+":")
  print(str(np.max(lip_EUvsDISINFO)))
  print("MAX EU_ENR_MIX4 | "+xai_method+":")
  print(str(np.max(lip_EU_ENR_MIX4)))
  print("MAX EMNAD_MIX3 | "+xai_method+":")
  print(str(np.max(lip_EU_EMNAD_MIX3)))

  #min lipschitz values
  print("MIN ISOT | "+xai_method+":")
  print(str(np.min(lip_ISOT)))
  print("MIN EUvsDISINFO | "+xai_method+":")
  print(str(np.min(lip_EUvsDISINFO)))
  print("MIN EU_ENR_MIX4 | "+xai_method+":")
  print(str(np.min(lip_EU_ENR_MIX4)))
  print("MIN EMNAD_MIX3 | "+xai_method+":")
  print(str(np.min(lip_EU_EMNAD_MIX3)))

  #print("ISOT lipschitz values all (sample of 10):")
  #print(str(lip_ISOT[:10]))

#'''

  print("-- Processing boxplots now. --")

  def box_plot_template(data, ylabel, legend, filename, title, xlabel, x_ax_range=None, y_ax_range=None, colors=None, dataset=None):
    plt.figure(figsize=(11.69, 8.27))
    if colors is None:
      colors = ['#ff0051', '#008bfb', '#2cb32c', '#ff8d00']
    if len(data) != len(legend):
      raise ValueError("The number of lines to be plotted must match the length of the legend list.")
    for i, d in enumerate(data):
      plt.boxplot(d, positions=[i], widths=0.6, showfliers=True, patch_artist=True,
                boxprops=dict(
                facecolor='white',
                edgecolor='black'),
                medianprops=dict(color=colors[i%len(colors)], linewidth=1.5),
                flierprops=dict(markeredgecolor=colors[i%len(colors)]#, alpha=0.25
                                ),
               # whiskerprops=dict(color=colors[i%len(colors)]),
               # capprops=dict(color=colors[i%len(colors)])
                )
      #plt.boxplot(d, positions=[i], widths=0.6, showfliers=True, patch_artist=False, medianprops=dict(color=colors[i%len(colors)], linewidth=2))
    #plt.rcParams["font.family"] = "Arial"  #findfont: Font family 'Arial' not found.
    plt.xlim(x_ax_range)
    plt.ylim(y_ax_range)
    plt.xticks(range(len(data)), legend , fontsize=22)
    plt.title(title, fontsize=22)
    #plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    #print("len data:")
    #print(len(data))
    if len(data) > 1:
      plt.axvline(x=0.5, color='#3b3b3b', linestyle=':')
    plt.grid(visible=True, which='both', axis='both', linestyle='--', color='gainsboro')
    #plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    #plt.savefig((path_data_created+filename), dpi=300, bbox_inches='tight')
    if xai_method == "Gradients":
      xai_method_save = "VANILLA_GRADIENTS"
    else:
      xai_method_save = xai_method
    ########################################################################
    #CHANGE EVAL SAVE SPACE WHEN DOING SLIGHT VS. SUBSTANTIAL PERTURBATIONS
    ########################################################################
    #NOTE: ALSO CHANGE INPUT DATA (5% vs. 20% perturbations)
    ########################################################################
    if path_perturbation_rate == 'perturbed_5percent/':
      #slight perturbations/eval3
      if dataset == "All" or dataset == "None":
        plt.savefig((store_path+'final_files/eval3/'+filename), dpi=300, bbox_inches='tight')
      else:
        plt.savefig((store_path+'final_files/eval3/'+dataset+'/'+xai_method_save+'/'+filename), dpi=300, bbox_inches='tight')
    elif path_perturbation_rate == 'perturbed_20percent/':
      #substantial perturbations/eval5
      if dataset == "All" or dataset == "None":
        plt.savefig((store_path+'final_files/eval5_subper/'+filename), dpi=300, bbox_inches='tight')
      else:
        plt.savefig((store_path+'final_files/eval5_subper/'+dataset+'/'+xai_method_save+'/'+filename), dpi=300, bbox_inches='tight')
    else:
      print("-- Error: No or wrong save space! --")
    #plt.show()

  data = [ lip_ISOT.flatten(), lip_EUvsDISINFO.flatten(), lip_EU_ENR_MIX4.flatten(), lip_EU_EMNAD_MIX3.flatten()]
  #legend = ['ISOT','EUvsDISINFO','EU_ENR_MIX4','EU_EMNAD_MIX3']
  legend = ['ISOT','EUvsDISINFO','ENR','EMNAD']
  ylabel = 'Lipschitz values'
  filename = 'lip_'+xai_method+'_All.png' #_All_zoom.png
  title = xai_method.replace("Gradients", "VANILLA GRADIENTS")+' | Lipschitz Constants | Perturbation Rate: '+''.join(c for c in path_perturbation_rate if c.isdigit())+'%'
  xlabel = 'x'
  box_plot_template(data, ylabel, legend, filename, title, xlabel, x_ax_range=None, y_ax_range=[0, 380], colors=None, dataset="All") #100 380
  
  data = [lip_ISOT.flatten()]
  legend = ['ISOT']
  ylabel = 'Lipschitz values'
  filename = 'lip_'+xai_method+'_ISOT.png'
  title = xai_method.replace("Gradients", "VANILLA GRADIENTS")+' | Lipschitz Constants - ISOT | Perturbation Rate: '+''.join(c for c in path_perturbation_rate if c.isdigit())+'%'
  xlabel = 'ISOT'
  box_plot_template(data, ylabel, legend, filename, title, xlabel, x_ax_range=None, y_ax_range=None, colors=['#008bfb'], dataset="ISOT")
  
  data = [lip_EUvsDISINFO.flatten()]
  legend = ['EUvsDISINFO']
  ylabel = 'Lipschitz values'
  filename = 'lip_'+xai_method+'_EUvsDISINFO.png'
  title = xai_method.replace("Gradients", "VANILLA GRADIENTS")+' | Lipschitz Constants - EUvsDISINFO | Perturbation Rate: '+''.join(c for c in path_perturbation_rate if c.isdigit())+'%'
  xlabel = 'EUvsDISINFO'
  box_plot_template(data, ylabel, legend, filename, title, xlabel, x_ax_range=None, y_ax_range=None, colors=['#008bfb'], dataset="EUvsDISINFO")

  data = [lip_EU_ENR_MIX4.flatten()]
  legend = ['EU_ENR_MIX4']
  ylabel = 'Lipschitz values'
  filename = 'lip_'+xai_method+'_EU_ENR_MIX4.png'
  title = xai_method.replace("Gradients", "VANILLA GRADIENTS")+' | Lipschitz Constants - ENR | Perturbation Rate: '+''.join(c for c in path_perturbation_rate if c.isdigit())+'%'
  xlabel = 'EU_ENR_MIX4'
  box_plot_template(data, ylabel, legend, filename, title, xlabel, x_ax_range=None, y_ax_range=None, colors=['#008bfb'], dataset="EU_ENR_MIX4")

  data = [lip_EU_EMNAD_MIX3.flatten()]
  legend = ['EU_EMNAD_MIX3']
  ylabel = 'Lipschitz values'
  filename = 'lip_'+xai_method+'_EU_EMNAD_MIX3.png'
  title = xai_method.replace("Gradients", "VANILLA GRADIENTS")+' | Lipschitz Constants - EMNAD | Perturbation Rate: '+''.join(c for c in path_perturbation_rate if c.isdigit())+'%'
  xlabel = 'EU_EMNAD_MIX3'
  box_plot_template(data, ylabel, legend, filename, title, xlabel, x_ax_range=None, y_ax_range=None, colors=['#008bfb'], dataset="EU_EMNAD_MIX3")

  print("-- All boxplots successfully processed and stored. --")

print("##### All done. #####")
