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
#model_name = "winterForestStump"

'''
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
'''
path_data_store='gdrive/benchmark/final_files/eval3/'

datasets_list = []

#If needed, split in in two parts and merge (to tackle I/O errors on Google Cloud / Drive - if any occur)

datasets=['ISOT', 'EUvsDISINFO']#, 'EU_ENR_MIX4', 'EU_EMNAD_MIX3']#
#datasets=['EU_ENR_MIX4', 'EU_EMNAD_MIX3']

load_datasets = ["gdrive/DATASETS/FINAL_DATASETS/ISOT_final.csv","gdrive/DATASETS/FINAL_DATASETS/jy46604790_fake_dataset_EU_final.csv"]#,"gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv","gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv"]
#load_datasets = ["gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv","gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv"]

#Loading dataset (only for testing)
#eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))
#df_data = pd.DataFrame(eval_data)
#df_data = df_data[["text", "label"]]
#print(df_data)

'''########################################################################################################################################################################################################################################################

cnt=0
for d in load_datasets:

    eval_data = load_dataset("csv", header=0, data_files=d, token=True, split="train").shuffle(seed=42).select(range(500))

    """# **Fidelity for Original Data **"""
    # (1) ISOT
    #eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/ISOT_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))                                   #ISOT (title+text)

    # (2) EUvsDISINFO data
    #eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/jy46604790_fake_dataset_EU_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

    # (3) ENR
    #MIX 4 [USE THIS]
    #eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

    # (4) EMNAD
    #MIX 3 [USE THIS]
    #eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

    df_data = pd.DataFrame(eval_data)
    df_data = df_data[["text", "label"]]
    print(df_data)

    eval_dataset = Dataset.from_pandas(df_data)
    print(eval_dataset)

    if d == 'gdrive/DATASETS/FINAL_DATASETS/ISOT_final.csv':
        tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect") #'roberta-base' = BASE MODEL  #OR: hamzab/roberta-fake-news-classification // both trained on ISOT
        model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
        model_name = "jy46604790"
    else:
        tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
        model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
        model_name = "winterForestStump"

    device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    task_evaluator = evaluator("text-classification")

    if model_name == "jy46604790":
        label_mapping_mode={"LABEL_0": 0, "LABEL_1": 1} #LABEL_0: Fake news // LABEL_1: Real news
    else: #"winterForestStump"
        label_mapping_mode={"FAKE": 0, "TRUE": 1}

    print("Label mapping mode:")
    print(str(label_mapping_mode))

    # 1. Pass a model name or path
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=eval_dataset,
        label_mapping=label_mapping_mode
    )

    # 2. Pass an instantiated model
    eval_results = task_evaluator.compute(
        model_or_pipeline=model,
        tokenizer=tokenizer,
        data=eval_dataset,
        label_mapping=label_mapping_mode
    )

    # 3. Pass an instantiated pipeline
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

    eval_results = task_evaluator.compute(
        model_or_pipeline=pipe,
        data=eval_dataset,
        metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
        label_mapping=label_mapping_mode,
        strategy="bootstrap",
        n_resamples=200 #confidence intervals
    )
    print("___________")
    print("MODEL: "+model_name+" | Original data" + " | Dataset: "+datasets[cnt])
    print("EVALUATION RESULTS:")
    print("___________")
    print(eval_results)

    ##########################################################
    #ADAPT
    ##########################################################
    datasets_list.append([datasets[cnt],0,eval_results['accuracy']['score'],eval_results['accuracy']['confidence_interval'],eval_results['recall']['score'],eval_results['recall']['confidence_interval'],eval_results['precision']['score'],eval_results['precision']['confidence_interval'],eval_results['f1']['score'], eval_results['f1']['confidence_interval']])

    cnt+=1

"""# **Fidelity for Slight Variations (check variations in performance based on accuracy)**"""

#path_perturbation_rate = 'perturbed_5percent/'
#path_perturbation_rate = 'perturbed_20percent/'

path_perturbation_rates=['perturbed_5percent/','perturbed_10percent/','perturbed_15percent/','perturbed_20percent/']


cnt_load_ds=0
for ds in datasets:

    eval_data = load_dataset("csv", header=0, data_files=load_datasets[cnt_load_ds], token=True, split="train").shuffle(seed=42).select(range(500))

    ##################################################
    #CHOOSE CORRECT ORIGINAL DATASET WITH FIXED SEED:
    ##################################################
    # (1) ISOT
    #eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/ISOT_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))                                   #ISOT (title+text)

    # (2) EUvsDISINFO data
    #eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/jy46604790_fake_dataset_EU_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

    # (3) ENR
    #MIX 4 [USE THIS]
    #eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

    # (4) EMNAD
    #MIX 3 [USE THIS]
    #eval_data = load_dataset("csv", header=0, data_files="gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv", token=True, split="train").shuffle(seed=42).select(range(500))

    df_data = pd.DataFrame(eval_data)
    df_data = df_data[["text", "label"]]
    print("ORIGINAL DATASET:")
    print(df_data)

    for path_perturbation_rate in path_perturbation_rates:

        #path_final=path+'final_files/'+dataset+'/'
        #path_perturbed=path_data_created+'perturbed/'
        path_perturbed='gdrive/benchmark/created_data/'+ds+'/'+path_perturbation_rate

        perturbed_texts = np.load(path_perturbed+ds+'_all_perturbed.npy', allow_pickle=True)

        print(str(len(perturbed_texts)))
        print("")
        #print("Samples perturbed texts:")
        #print(perturbed_texts[:10])

        perturbed_texts_eval_data=[]

        i = 0
        while i < len(perturbed_texts):
            perturbed_texts_eval_data.append(perturbed_texts[i][0]) #first sample of perturbed texts
            i+=1

        print(str(len(perturbed_texts_eval_data)))
        print("")

        print("First three samples of perturbed texts:")
        print(perturbed_texts_eval_data[:3])

        print("")
        print("Original eval dataframe text: ")
        print(df_data.iloc[0])

        i = 0
        while i < len(perturbed_texts_eval_data):
            #df_data["text"].iloc[i] = perturbed_texts_eval_data[i] #triggers warning, use instead: (see below)
            df_data.loc[i, "text"] = perturbed_texts_eval_data[i]
            i+=1

        print("")
        print(("Perturbed eval dataframe text: "))
        print(df_data.iloc[0])

        print("Dataframe with perturbed texts:")
        print(df_data)

        #convert df to dataset object
        eval_dataset = Dataset.from_pandas(df_data)
        print(eval_dataset)

        print("Final data sent to evaluator:")
        df = pd.DataFrame(eval_dataset)
        df = df[["text", "label"]]
        print(df)

        #use small perturbations as above - run models over it - compare and analyze prediction output/performance based on performance metrics

        if ds == 'ISOT':
            tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect") #'roberta-base' = BASE MODEL  #OR: hamzab/roberta-fake-news-classification // both trained on ISOT
            model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
            model_name='jy46604790'
        else:
            tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
            model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
            model_name='winterForestStump'


        device = torch.device(str("cuda:0") if torch.cuda.is_available() else "cpu")
        model = model.to(device)

        task_evaluator = evaluator("text-classification")

        if model_name == "jy46604790":
            label_mapping_mode={"LABEL_0": 0, "LABEL_1": 1} #LABEL_0: Fake news // LABEL_1: Real news
        else: #"winterForestStump"
            label_mapping_mode={"FAKE": 0, "TRUE": 1}

        print("Label mapping mode:")
        print(str(label_mapping_mode))

        # 1. Pass a model name or path
        eval_results = task_evaluator.compute(
            model_or_pipeline=model,
            tokenizer=tokenizer,
            data=eval_dataset,
            label_mapping=label_mapping_mode
        )

        # 2. Pass an instantiated model
        eval_results = task_evaluator.compute(
            model_or_pipeline=model,
            tokenizer=tokenizer,
            data=eval_dataset,
            label_mapping=label_mapping_mode
        )

        # 3. Pass an instantiated pipeline
        pipe = pipeline("text-classification", model=model, tokenizer=tokenizer)

        eval_results = task_evaluator.compute(
            model_or_pipeline=pipe,
            data=eval_dataset,
            metric=evaluate.combine(["accuracy", "recall", "precision", "f1"]),
            label_mapping=label_mapping_mode,
            strategy="bootstrap",
            n_resamples=200 #confidence intervals
        )
        print("___________")
        print("MODEL: "+model_name+" | Perturbation rate: "+path_perturbation_rate + " | Dataset: "+ds)
        print("EVALUATION RESULTS:")
        print("___________")
        print(eval_results)

        datasets_list.append([ds,((int)("".join([ele for ele in path_perturbation_rate if ele.isdigit()]))),eval_results['accuracy']['score'],eval_results['accuracy']['confidence_interval'],eval_results['recall']['score'],eval_results['recall']['confidence_interval'],eval_results['precision']['score'],eval_results['precision']['confidence_interval'],eval_results['f1']['score'], eval_results['f1']['confidence_interval']]) #dataset name/perturbation rate/F1 score

    #example
    #data = [['ISOT', 5, 0.978], ['ISOT', 5, 0.978], ['ISOT', 5, 0.978]]
    df_fid = pd.DataFrame(datasets_list, columns=['name_dataset', 'perturbation_rate', 'accuracy', 'a_ci', 'recall', 'r_ci', 'precision', 'p_ci', 'f1_score', 'f_ci'])
    print(df_fid)

    #store and overwrite after each dataset run
    df_fid.to_csv(path_data_store+'PART_2'+'_final_fidelity.csv') #PART_1 and PART_2 due to Google I/O error splitting generating metrics
    print("-- Saved "+ds+"! --")

    cnt_load_ds+=1

#example
#data = [['ISOT', 5, 0.978], ['ISOT', 5, 0.978], ['ISOT', 5, 0.978]]
#df_fid = pd.DataFrame(datasets_list, columns=['name_dataset', 'perturbation_rate', 'accuracy', 'recall', 'precision', 'f1_score'])

#print(df_fid)

#df_fid.to_csv(path_data_store+'final_fidelity.csv')

print("-- Saved all! --")

'''########################################################################################################################################################################################################################################################

'''########################################################################################################################################################################################################################################################

print("Load data...")

df_fid_1 = pd.read_csv(path_data_store+'PART_1_final_fidelity.csv')
df_fid_2 = pd.read_csv(path_data_store+'PART_2_final_fidelity.csv')

#merge files
df_fid = pd.concat([df_fid_1, df_fid_2])

df_fid.to_csv(path_data_store+'final_fidelity.csv')

df_fid.to_excel(path_data_store+"final_fidelity.xlsx")

print("Stored!")

'''########################################################################################################################################################################################################################################################

#'''########################################################################################################################################################################################################################################################

print("Load data...")

df_fid = pd.read_csv(path_data_store+'final_fidelity.csv')

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(df_fid)

#drop some metrics
df_fid = df_fid.drop('accuracy', axis=1)
df_fid = df_fid.drop('recall', axis=1)
df_fid = df_fid.drop('precision', axis=1)
df_fid = df_fid.drop('a_ci', axis=1)
df_fid = df_fid.drop('r_ci', axis=1)
df_fid = df_fid.drop('p_ci', axis=1)
df_fid = df_fid.drop('f_ci', axis=1)
df_fid = df_fid.drop('Unnamed: 0', axis=1)
df_fid = df_fid.drop('Unnamed: 0.1', axis=1)
#df_fid = df_fid.drop('name_dataset', axis=1)

df_fid = df_fid.sort_values("perturbation_rate")

#'''########################################################################################################################################################################################################################################################

#with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#    print(df_fid)

#'''########################################################################################################################################################################################################################################################

datasets=['ISOT', 'EUvsDISINFO', 'EU_ENR_MIX4', 'EU_EMNAD_MIX3']
#load_datasets = ["gdrive/DATASETS/FINAL_DATASETS/ISOT_final.csv","gdrive/DATASETS/FINAL_DATASETS/jy46604790_fake_dataset_EU_final.csv","gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_4_enr_final.csv","gdrive/DATASETS/FINAL_DATASETS/EU_FINAL_DATASETS/EU_mix_3_EMNAD_final.csv"]

#save results in the format
#df[DS] = [[x=pert_rate,y=F1 score],[x=pert_rate,y=F1 score],[x=pert_rate,y=F1 score],[x=pert_rate,y=F1 score],[x=pert_rate,y=F1 score]]
#ds=[] #4 datasets and each is a df with the data that is plotted then

import matplotlib.pyplot as plt

#plot F1 scores
def line_plot_template(data, ylabel, legend, filename, title, xlabel, x, x_ax_range=None, y_ax_range=None, colors=None, markers=None):
    plt.figure(figsize=(11.69, 8.27))
    if colors is None:
        # colors: red, blue, green,  Burgundy, Navy blue, Olive Drab, gold, Dark khaki ,'#ffd700', '#cfb53b'
        # colors:
        #colors = ['#ff0051', '#008bfb', '#2cb32c', '#ff0051', '#008bfb', '#2cb32c', '#8b0000', '#00008b']#, '#134d13']
        colors = ['#f91010', '#0456f8', '#00e507', '#a504f1']
    if markers is None:
        markers = ['o','*','x','_']
    if len(data) != len(legend):
        raise ValueError("The number of lines to be plotted must match the length of the legend list.")
    for i, d in enumerate(data):
      plt.plot(x, d, color=colors[i], marker=markers[i])
    plt.rcParams["font.family"] = "Arial"
    plt.xlim(x_ax_range)
    plt.ylim(y_ax_range)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    #plt.legend(legend, fontsize=14)
    plt.legend(legend, fontsize=14, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.grid(visible=True, which='both', axis='both', linestyle='--', color='gainsboro')
    plt.savefig((path_data_store+filename), dpi=300, bbox_inches='tight')
    #plt.show()

df_fid_final = {} #create dictionary

for dataset_n in datasets:
    select = df_fid[ (df_fid['name_dataset'] == dataset_n) ]
    select = select.drop('name_dataset', axis=1)    #perturbation rate/F1 score left, drop dataset name
    #print(select)
    df_fid_final[dataset_n] = select

#for x in df_fid_final:
    #print(df_fid_final[x])

#'''########################################################################################################################################################################################################################################################

#'''########################################################################################################################################################################################################################################################
#plot
cols=['f1_score']
data = [df_fid_final[ds][col] for col in cols for ds in datasets ]
x = df_fid_final['ISOT']['perturbation_rate']
legend = ['ISOT', 'EUvsDISINFO', 'EU_ENR_MIX4', 'EU_EMNAD_MIX3']
ylabel = 'F1 score'
filename = 'final_fidelity.png'
title = 'Fidelity for Input Variations'
xlabel = 'Perturbation rate'
line_plot_template(data, ylabel, legend, filename, title, xlabel, x,)

#'''########################################################################################################################################################################################################################################################