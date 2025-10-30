# -*- coding: utf-8 -*-

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

#login to hugging face
!huggingface-cli login --token [TOKEN] #--add-to-git-credential

from transformers import pipeline
from transformers import AutoModelForSequenceClassification, pipeline
from datasets import Dataset
from transformers import AutoTokenizer
from datasets import load_dataset
from evaluate import evaluator
import evaluate

path='/content/drive/MyDrive/benchmark/'

# !!! SET MODEL NAME - IMPORTANT !!! #

# Set model - for ISOT the first, for all other datasets the second

#model_name = 'jy46604790'

model_name = 'winterForestStump'

# !!! SET MODEL NAME - IMPORTANT !!! #

# Define paths
'''
#ISOT
dataset='ISOT'
path_data_created='/content/drive/MyDrive/benchmark/created_data/ISOT/'
path_csv= path_data_created+model_name+'_ISOT_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_ISOT.sav'
lime_path=path_data_created+model_name+'_lime_values_500_ISOT.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_ISOT.sav'
#outputs_path=path_data_created+model_name+'_original_probs_ISOT_500.pt'
outputs_path=path_data_created+model_name+'_original_probs_v2_ISOT_500'
path_labels=path_data_created+model_name+'_labels_ISOT_500.npy'

#EUvsDISINFO
dataset='EUvsDISINFO'
path_data_created='/content/drive/MyDrive/benchmark/created_data/EUvsDISINFO/'
path_csv= path_data_created+model_name+'_EUvsDISINFO_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EUvsDISINFO.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EUvsDISINFO.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_EUvsDISINFO.sav'
#outputs_path=path_data_created+model_name+'_original_probs_EUvsDISINFO_500.pt'
outputs_path=path_data_created+model_name+'_original_probs_v2_EUvsDISINFO_500'
path_labels=path_data_created+model_name+'_labels_EUvsDISINFO_500.npy'

#ENR MIX 4
dataset='EU_ENR_MIX4'
path_data_created='/content/drive/MyDrive/benchmark/created_data/EU_ENR_MIX4/'
path_csv= path_data_created+model_name+'_EU_ENR_MIX4_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EU_ENR_MIX4.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EU_ENR_MIX4.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_EU_ENR_MIX4.sav'
#outputs_path=path_data_created+model_name+'_original_probs_EU_ENR_MIX4_500.pt'
outputs_path=path_data_created+model_name+'_original_probs_v2_EU_ENR_MIX4_500'
path_labels=path_data_created+model_name+'_labels_EU_ENR_MIX4_500.npy'
'''
#EMNAD MIX 3
dataset='EU_EMNAD_MIX3'
path_data_created='/content/drive/MyDrive/benchmark/created_data/EU_EMNAD_MIX3/'
path_csv= path_data_created+model_name+'_EU_EMNAD_MIX3_500.csv'
shap_path=path_data_created+model_name+'_shap_values_500_EU_EMNAD_MIX3.sav'
lime_path=path_data_created+model_name+'_lime_values_500_EU_EMNAD_MIX3.sav'
gradient_path=path_data_created+model_name+'_gradient_values_v2_500_EU_EMNAD_MIX3.sav'
#outputs_path=path_data_created+model_name+'_original_probs_EU_EMNAD_MIX3_500.pt'
outputs_path=path_data_created+model_name+'_original_probs_v2_EU_EMNAD_MIX3_500'
path_labels=path_data_created+model_name+'_labels_EU_EMNAD_MIX3_500.npy'


#xai_method_name = "SHAP"
xai_method_name = "LIME"
#xai_method_name = "VANILLA_GRADIENTS"

#UPDATE outputs path according to XAI method combination with dataset (slight variations in tokenization and word-level handling)
outputs_path = outputs_path+'_'+xai_method_name+'.pt'

#Run LIME or gradient instead of SHAP (data input structure is modified and the same):
shap_path = lime_path
#shap_path = gradient_path

if dataset == 'ISOT':
  tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect") #'roberta-base' = BASE MODEL  #OR: hamzab/roberta-fake-news-classification // both trained on ISOT
  model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
  model_name = 'jy46604790'
else:
  tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
  model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
  model_name = 'winterForestStump'
'''
elif dataset == 'EUvsDISINFO':
  tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
  model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
elif dataset == 'EU_ENR_MIX4':
  tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
  model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
elif dataset == 'EU_EMNAD_MIX3':
  tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
  model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
else:
  tokenizer = RobertaTokenizer.from_pretrained("FacebookAI/roberta-base")
  model = AutoModelForSequenceClassification.from_pretrained("FacebookAI/roberta-base")
  #tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
  #model = AutoModelForSequenceClassification.from_pretrained("google-bert/bert-base-uncased")
'''

'''
# # # # # # MODEL # # # # # #
# Instantiate the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("jy46604790/Fake-News-Bert-Detect")
model = AutoModelForSequenceClassification.from_pretrained("jy46604790/Fake-News-Bert-Detect")
model_name = 'jy46604790'

#fine-tuned model on EU disinformation
#tokenizer = AutoTokenizer.from_pretrained("winterForestStump/Roberta-fake-news-detector")
#model = AutoModelForSequenceClassification.from_pretrained("winterForestStump/Roberta-fake-news-detector")
#model_name = 'winterForestStump'
'''

"""# loading the dataframe, the labels, the shap values and the original probabilities


"""

path_final=path+'final_files/eval1/'+dataset+'/'+xai_method_name+'/'
df = pd.read_csv(path_csv)
labels = np.load(path_labels)

#DATA SAMPLE text and labels
#eval_data = load_dataset("csv", header=0, data_files="/content/drive/MyDrive/DATASETS/FINAL_DATASETS/"+dataset+"_final_only_title.csv", token=True, split="train").shuffle(seed=42).select(range(10))  #ISOT (only title)
#df = pd.DataFrame(eval_data)
#df = df[["text", "label"]]

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

#LOAD SHAP values and original probs
if xai_method_name=="SHAP":
  shap_values = pickle.load(open(shap_path, 'rb'))
elif xai_method_name=="LIME":
  shap_values = pickle.load(open(lime_path, 'rb'))
elif xai_method_name=="VANILLA_GRADIENTS":
  shap_values = pickle.load(open(gradient_path, 'rb'))
'''
#shap_values = pickle.load(open(path_data_created+'jy46604790'+'_lime_values_500_ISOT.sav', 'rb'))
#shap_values = pickle.load(open(path_data_created+'jy46604790_gradient_values_500_ISOT.sav', 'rb'))
shap_values = pickle.load(open(path_data_created+'jy46604790_shap_values_500_ISOT.sav', 'rb'))
print(len(shap_values))
'''

original_probs= torch.load(outputs_path, map_location=torch.device('cpu'))
labels=torch.LongTensor(labels)

#sm = torch.nn.Softmax(dim=1)
#probs_original = sm(original_probs)
#print(str(probs_original))


if xai_method_name != "SHAP":
  shap_values_final = SHAPVals()
  for i in shap_values:
    shap_values_final.data.append(i.data)
    shap_values_final.values.append(i.values)
  shap_values = shap_values_final


'''
#Check data
#print(str(len(shap_values.data)))  #len = 1, should be 500; issue with savind re-structured values, gradients are fine BUT LIME not!
print(len(shap_values.data[80]))
print(shap_values.data[80])
print(len(shap_values.values[80]))
print(shap_values.values[80])
print(shap_values.data[1][1] + " is the string")
print(str(shap_values.values[1][1]) + " is the weight")
#print(shap_values.data[1][1])
#print(shap_values.values[1][1])
'''

#check for encoding issues of Gradients and replace them with ""
#i=0
#while i < 20:
  #print(shap_values.data[i])
  #i+=1
  #shap_values.data[2] = [s.replace("Ċ", "") for s in shap_values.data[2]]
  #shap_values.data[2] = [s.replace("âĢĵ", "") for s in shap_values.data[2]]

print("XAI METHOD DATA AND VALUES[2]: ")
print(shap_values.data[2])
print(shap_values.values[2])
print(len(shap_values.data))
print(len(shap_values.values))

#print(shap_values.data[:1]) #for re-structure access via ==> shap_values[0].data[:5]

#display(df.to_string())
print(df)

print(labels)

"""# **ONLY FOR LIME - APPLY Zero-Weights to unweighted features (tokens)**"""

import nltk
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

if xai_method_name == "LIME":

  lime_obj_all = []
  texts = []
  i=0
  while i < len(pd.read_csv(path_csv)):
    texts.append(df.iloc[i].text)
    i+=1
  print(len(texts))

  for t in texts:
    #encoded_tokens = tokenizer.tokenize(t, max_length=512, truncation=True)  #LIME returns full words with weights so tokenization with splitting up words in tokens doesn't work
    encoded_tokens = nltk.tokenize.word_tokenize(t)[:512]
    #char = 'Ġ'
    #for idx, ele in enumerate(encoded_tokens):
    #  encoded_tokens[idx] = ele.replace(char, '')

    #print(encoded_tokens)
    #lime_obj = LimeVals(np.zeros(len(encoded_tokens)), encoded_tokens)
    #encoded_tokens = tokenizer.tokenize(text, max_length=512, truncation=True)
    #lime_obj.data.append(encoded_tokens)
    #lime_obj.values.append(np.zeros(len(encoded_tokens)))
    lime_obj_all.append(LimeVals(np.zeros(len(encoded_tokens)), encoded_tokens))

  print(lime_obj_all[2].values)
  print(lime_obj_all[2].data)
  print(len(lime_obj_all[2].data)) #27

  print(shap_values.values[2])
  print(shap_values.data[2])
  print(len(shap_values.data[2])) #22

  print("Sanity check (should be equal):")
  print(len(lime_obj_all))
  print(len(shap_values.values))

  cnt=0
  while cnt < len(lime_obj_all):  #len=500 instances
    data = shap_values.data[cnt]
    values = shap_values.values[cnt]
    data_1 = lime_obj_all[cnt].data
    values_1 = lime_obj_all[cnt].values
    cnt_emb=0
    while cnt_emb < len(data_1): #e.g. 27
      cnt_dv=0
      while cnt_dv < len(data): #e.g. 22
        if data_1[cnt_emb] == data[cnt_dv]:
          values_1[cnt_emb] = values[cnt_dv] #if word/token matches, then apply LIME value, otherwise it is zero; keeps also correct order of tokens
        cnt_dv+=1
      cnt_emb+=1
    shap_values.data[cnt] = data_1
    shap_values.values[cnt] = values_1
    cnt+=1

  print("LEN shap_values.data[0] and .values[0]:")
  print(str(len(shap_values.data[0])))
  print(str(len(shap_values.values[0])))
  print("Detailed data and values[0]:")
  print(str(shap_values.data[0]))
  print(str(shap_values.values[0]))
  print("LEN shap_values.data and .values:")
  print(str(len(shap_values.data)))
  print(str(len(shap_values.values)))

  print("Example instance 2:")
  print(shap_values.data[2])
  print(shap_values.values[2])
  print("LEN example instance 2:")
  print(len(shap_values.data[2]))
  print(len(shap_values.values[2]))

  #for ratio calculation later only LIME values with weight used, not zero values

"""# **definitions**"""

def encode_df(data, tokenizer, labels):
    data = data.fillna("")
    bert_encoded_dict = data.apply(lambda x: tokenizer.encode_plus("".join(x).lstrip().replace(" .", ".").replace(" , ", ", "),
                      add_special_tokens=True, # Add '[CLS]' and '[SEP]'
                      max_length = 512,           # Pad & truncate all sentences. #max length of model = 512 #or use tokenizer.model_max_length
                      padding='max_length',
                      return_attention_mask = True,   # Construct attn. masks.
                      return_tensors = 'pt',     # Return pytorch tensors.
                      truncation = True
                      ),axis=1)
    bert_input_ids = torch.cat([item['input_ids'] for item in bert_encoded_dict], dim=0)
    bert_attention_masks = torch.cat([item['attention_mask'] for item in bert_encoded_dict], dim=0)
    # Format targets
    labels = torch.tensor(labels)
    return {'input_ids':bert_input_ids, 'attention_masks':bert_attention_masks, 'labels':labels}

#get_global_feature_list
#based on suggestions from https://github.com/slundberg/shap/issues/632
def get_global_feature_list(shap_values):
    data = list(chain.from_iterable(shap_values.data))
    values = list(chain.from_iterable(shap_values.values))
    df = pd.DataFrame({'data': data, 'values': values})
    df['values'] = abs(df['values'])
    df_grouped = df.groupby('data', as_index=False).mean().sort_values('values',ascending=False)
    return df_grouped

#deletion and preservation check implementation based on https://github.com/Jianbo-Lab/L2X/tree/master/imdb-sent
def replace_global_highest_elements(shap_values, k, tokenizer, labels, replacement_type):
    global_features = get_global_feature_list(shap_values)
    # gets the features/words from the shap value object
    data = pd.DataFrame(shap_values.data)
    data_rand=data.copy()
    if replacement_type == "top":
        # takes the top k% of the global features or random
        replacement_features = global_features.head(int(len(global_features)*k))
        replacement_features_rand = global_features['data'].sample(frac=k)

    elif replacement_type == "bottom":
        # takes the bottom 100-k% of the global features or random
        replacement_features = global_features.tail(int(len(global_features)*(1-k)))
        replacement_features_rand = global_features['data'].sample(frac=(1-k))

    if k != 0.01: #k=0.01 for BASELINE generation
      data = data.where(~data.isin(replacement_features["data"].tolist()), "[PAD]")
      data_rand = data_rand.where(~data_rand.isin(replacement_features_rand.tolist()), "[PAD]")
    elif k == 0.01:
      print("BASELINE processing...")

    en_replaced=encode_df(data, tokenizer, labels)
    en_rand=encode_df(data_rand, tokenizer, labels)
    return en_replaced, en_rand

def replace_k_highest_random_elements(shap_values, k, tokenizer, labels, replacement_type):
    print('replace_local')
    replaced_selected=[]; replaced_rand=[]
    for i in range(len(shap_values.values)):
      if i==10:
        print('k before if: '+str(k))
      if len(shap_values.values[i]) < k:
        lk=len(shap_values.values[i])
        ind_high= (np.argsort(np.abs(shap_values.values[i]))[::-1]) [:lk]
        ind_rand = random.sample(range(len(shap_values.values[i])), lk)
      else:
        #replace highest
        ind_high= (np.argsort(np.abs(shap_values.values[i]))[::-1]) [:k]
        ind_rand = random.sample(range(len(shap_values.values[i])), k)
      l= shap_values.data[i]
      if replacement_type=='top':
        selected = ["[PAD]" if np.isin(i, ind_high) else x for i, x in enumerate(l)]
        rand = ["[PAD]" if np.isin(i, ind_rand) else x for i, x in enumerate(l)]

      elif replacement_type=='bottom':
        selected = ["[PAD]" if not np.isin(i, ind_high) else x for i, x in enumerate(l)]
        rand = ["[PAD]" if not np.isin(i, ind_rand) else x for i, x in enumerate(l)]
        if i==10:
          print('k after if: '+str(k))
          #print(selected)
          #print('ind_high:'+str(ind_high))
          print('ind_high:'+str(len(ind_high)))

      replaced_selected.append(selected)
      replaced_rand.append(rand)

    data=pd.DataFrame(replaced_selected); data_rand=pd.DataFrame(replaced_rand)
    en_replaced= encode_df(data, tokenizer, labels)
    en_rand= encode_df(data_rand, tokenizer, labels)
    return en_replaced, en_rand

def compute_acc_and_lor(shap_values, labels, k, tokenizer, model, original_prob, device, replacement_area, replacement_type):
    metrics = pd.DataFrame(columns=['acc_selected', 'flip_selected', 'lor_change_selected', 'apoc_selected', 'acc_rand', 'flip_rand', 'lor_change_rand', 'apoc_rand'])
    print('k in compute:'+str(k))
    if replacement_area == 'global':
        en_replaced, en_rand = replace_global_highest_elements(shap_values,
            k, tokenizer, labels, replacement_type)
    elif replacement_area =='local':
        en_replaced, en_rand = replace_k_highest_random_elements(shap_values,
            k, tokenizer, labels, replacement_type=replacement_type)

    with torch.no_grad():
      model.to(device)
      p_selected = model(en_replaced['input_ids'].to(device),
                         attention_mask=en_replaced['attention_masks'].to(device), labels=en_replaced['labels'].to(device))
      p_rand = model(en_rand['input_ids'].to(device),
                     attention_mask=en_rand['attention_masks'].to(device), labels=en_rand['labels'].to(device))
    #https://github.com/icrto/xML/issues/1
    probs_selected = F.softmax(p_selected.logits, dim=1)
    probs_rand = F.softmax(p_rand.logits, dim=1)
    probs_original = F.softmax(original_prob, dim=1)

    for i in range(len(shap_values.values)):
        x_selected = probs_selected[i].cpu(); prob = probs_original[i].cpu()
        x_rand = probs_rand[i].cpu()
        label = labels[i].cpu()
        cls = np.argmax(prob)#.item() #returns highest prob as int
        acc_selected = torch.eq(label, torch.argmax(x_selected))
        flip_selected = np.argmax(prob)==np.argmax(x_selected)
        #lor based on https://github.com/Jianbo-Lab/LCShapley/blob/master/texts/utils.py
        lor_change_selected = compute_lor(x_selected, cls) - compute_lor(prob, cls)
        aopc_selected = AOPC(prob, x_selected, cls)
        acc_rand = torch.eq(label, torch.argmax(x_rand))
        flip_rand = np.argmax(prob)==np.argmax(x_rand)
        lor_change_rand = compute_lor(x_rand, cls) - compute_lor(prob, cls)
        aopc_rand = AOPC(prob, x_rand, cls)
        #metrics = metrics.append({'acc_selected':acc_selected,'flip_selected':flip_selected, 'lor_change_selected':lor_change_selected, 'aopc_selected':aopc_selected, 'acc_rand':acc_rand, 'flip_rand':flip_rand, 'lor_change_rand':lor_change_rand, 'aopc_rand':aopc_rand}, ignore_index=True)
        metrics = metrics._append({'acc_selected':acc_selected,'flip_selected':flip_selected, 'lor_change_selected':lor_change_selected, 'aopc_selected':aopc_selected, 'acc_rand':acc_rand, 'flip_rand':flip_rand, 'lor_change_rand':lor_change_rand, 'aopc_rand':aopc_rand}, ignore_index=True) #append deprecated, use _append or concat

    avg_acc_selected = np.mean(metrics['acc_selected'].astype(float).values)
    avg_flip_selected = np.mean(metrics['flip_selected'].astype(float).values)
    avg_lor_selected = metrics['lor_change_selected'].mean()
    avg_aopc_selected = metrics['aopc_selected'].mean()
    avg_acc_rand = np.mean(metrics['acc_rand'].astype(float).values)
    avg_flip_rand = np.mean(metrics['flip_rand'].astype(float).values)
    avg_lor_rand = metrics['lor_change_rand'].mean()
    avg_aopc_rand = np.mean(metrics['aopc_rand'].astype(float).values)
    return_metrics ={'k':k, 'acc_selected':avg_acc_selected,'flip_selected':avg_flip_selected,
    'lor_change_selected':avg_lor_selected,'aopc_selected':avg_aopc_selected, 'acc_rand':avg_acc_rand, 'flip_rand':avg_flip_rand,
    'lor_change_rand':avg_lor_rand, 'aopc_rand':avg_aopc_rand}
    return return_metrics



def compute_acc_and_lor_BASE(shap_values, labels, k, tokenizer, model, device, replacement_area, replacement_type):
  print('k in compute:'+str(k))
  if replacement_area == 'global':
      en_replaced, en_rand = replace_global_highest_elements(shap_values,
          k, tokenizer, labels, replacement_type)
  with torch.no_grad():
    model.to(device)
    p_base = model(en_replaced['input_ids'].to(device),
                         attention_mask=en_replaced['attention_masks'].to(device), labels=en_replaced['labels'].to(device))
  probs_base = p_base.logits
  torch.save(probs_base,outputs_path)

def replace_and_get_accuracy_flip_lor_BASE(shap_values, tokenizer, model, labels, device, replacement_area, replacement_type,  k_range = [0.0], ratio = 'off'):
    for k in k_range:
      print('k in replace:'+str(k))
      compute_acc_and_lor_BASE(shap_values, labels, k, tokenizer, model, device, replacement_area, replacement_type)



def replace_and_get_accuracy_flip_lor(shap_values, tokenizer, model, labels, original_prob, device, replacement_area, replacement_type,  k_range = np.arange(0, 0.21, 0.01), ratio = 'off'):

  accuracy_df = pd.DataFrame(columns =['k', 'acc_selected', 'flip_selected', 'lor_change_selected', 'aopc_selected', 'acc_rand', 'flip_rand', 'lor_change_rand', 'aopc_rand'])

  if ratio == 'on':
    cnt=0
    len_all = len(shap_values.values)
    len_list_single_ratios = []
    avg_ratio_all = 0
    #print("LEN ALL:")
    #print(str(len_all))

    #LIME - BEGIN
    if xai_method_name == "LIME": #For LIME do not use zero values for ratio calculation
      while cnt < len_all:
        #print("LEN before removing zeros:")
        #print(shap_values.data[cnt])
        #print(shap_values.values[cnt])
        #print(len(shap_values.values[cnt]))
        #print("LEN after removing zeros:")
        #print(list(filter(lambda num: num != 0, shap_values.data[cnt])))
        #print(list(filter(lambda num: num != 0, shap_values.values[cnt])))
        #print(len(list(filter(lambda num: num != 0, shap_values.values[cnt]))))
        #print("######################################")
        if len(list(filter(lambda num: num != 0, shap_values.values[cnt]))) > 512:
          len_list_single_ratios.append(512)
        else:
          len_list_single_ratios.append(len(list(filter(lambda num: num != 0, shap_values.values[cnt]))))
        cnt+=1
    else:
      while cnt < len_all:
        if len(shap_values.values[cnt]) > 512:
          len_list_single_ratios.append(512)
        else:
          len_list_single_ratios.append(len(shap_values.values[cnt]))
        cnt+=1
      #print("Example values per instance:")
      #print(shap_values.values[cnt])
    #LIME - END

    #while cnt < len_all:
    #  if len(shap_values.values[cnt]) > 512:
    #    len_list_single_ratios.append(512)
    #  else:
    #    len_list_single_ratios.append(len(shap_values.values[cnt]))
    #  cnt+=1
      #print("Example values per instance:")
      #print(shap_values.values[cnt])
    print("List of all single ratios per instance:")
    print(str(len_list_single_ratios))
    print("Len list single ratios or overall instances:")
    print(str(len(len_list_single_ratios)))
    avg_ratio_all = np.mean(len_list_single_ratios)
    #print("AVG RATIO ALL: ")
    #print(str(avg_ratio_all))
    print("20% of AVG RATIO ALL:")
    print(str(round(avg_ratio_all/100*20)))
    k_range = np.arange(0, round(avg_ratio_all/100*20), 1)
    #print(str(k_range)

  #print("#############################")
  #print("k_range in use is: ")
  #print(str(k_range))
  #print("#############################")

  for k in k_range:
    print('k in replace:'+str(k))
    #accuracy_df = accuracy_df.append(pd.DataFrame.from_dict(compute_acc_and_lor(shap_values, labels, k, tokenizer, model, original_prob, device, replacement_area, replacement_type), orient='index').transpose()) #append deprecated, use _append or concat
    accuracy_df = pd.concat([accuracy_df, pd.DataFrame.from_dict(compute_acc_and_lor(shap_values, labels, k, tokenizer, model, original_prob, device, replacement_area, replacement_type), orient='index').transpose()], ignore_index=True)
    #accuracy_df = accuracy_df._append(pd.DataFrame.from_dict(compute_acc_and_lor(shap_values, labels, k, tokenizer, model, original_prob, device, replacement_area, replacement_type), orient='index').transpose())
  return accuracy_df



def compute_lor(prob, cls):
	return np.log(prob[cls] + 1e-6) - np.log(1 - prob[cls] + 1e-6) #Laplace smoothing or additive smoothing (adding: + 1e-6)

def AOPC(original_probs, modified_probs, cls):
    return (original_probs[cls]-modified_probs[cls]).item()

import matplotlib.pyplot as plt

def line_plot_template( y, z,  ylabel, legend, filename, title, xlabel, x,):
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, color='blue', marker='o')
    plt.plot(x, z, color='red', marker='o')
    plt.rcParams["font.family"] = "Arial"
    plt.title(title)
    plt.xlabel("k", fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(legend, fontsize=14)
    plt.grid(visible=True, which='both', axis='both', linestyle='--', color='grey')
    plt.savefig((path_final+filename), dpi=300, bbox_inches='tight')
    plt.show()

"""# **BASELINE**"""

#compute_acc_baseline(shap_values, tokenizer, model, labels, device, 'global', 'top',  outputs_path)

#def compute_acc_baseline(shap_values, labels, tokenizer, model, device, replacement_area, replacement_type, outputs_path): #FIRST ROUND WITH NO CHANGES IN DELETION OR ANY OTHER VARIATION - get baseline performance and accuracy
#print('BASELINE k in compute:'+str(0))
#if replacement_area == 'global':

#en_replaced, en_rand = replace_global_highest_elements(shap_values, 0, tokenizer, labels, replacement_type='top')

#with torch.no_grad():
#    model.to(device)

#p_baseline = model(en_replaced['input_ids'].to(device),attention_mask=en_replaced['attention_masks'].to(device), labels=en_replaced['labels'].to(device))
#probs_baseline = p_baseline.logits
#print(probs_baseline)
#torch.save(probs_selected,outputs_path)

replace_and_get_accuracy_flip_lor_BASE(shap_values, tokenizer, model, labels, device, 'global', 'top',  k_range = [0.01])

print("--- BASELINE stored. ---")

#LOAD baseline
#original_probs= torch.load(outputs_path, map_location=torch.device('cpu'))
#print(original_probs[:10])
#len(original_probs)

"""# **metrics**"""

from sklearn.metrics import classification_report

report = classification_report(labels, torch.argmax(original_probs, axis=1), output_dict=True)
df_report = pd.DataFrame(report).transpose()
df_report.to_latex(path_data_created+dataset+'_report.tex')

#average statement token length
length_tokens=[]
for i in range(len(shap_values.data)):
  length_tokens.append(len(shap_values.data[i]))
np.mean(length_tokens)

#numbers of unique tokens
global_list = get_global_feature_list(shap_values)
#numbers of total tokens
doubles_list = data = list(chain.from_iterable(shap_values.data))
#distribution of labels
print(np.mean(labels.numpy()))

df_description = pd.DataFrame({'Unique Tokens': [len(global_list)],
                  'Total Tokens': [len(doubles_list)],
                    'Avg Token per Statement': [np.mean(length_tokens)],
                  'Label Distribution': [np.mean(labels.numpy())]})
df_description.to_latex(path_data_created+dataset+'_token.tex')

df_description

#np.random.seed(42)
#rand_index= np.random.randint(0, high=500, size=5) #put sample size here, currently: 10 or 1000 // sample size = 500
#sample = df.loc[rand_index].copy()
#with pd.option_context("max_colwidth", 500): #sample size 500
#    print(sample.to_latex(path_data_created+dataset+'_sample.tex'))

"""# **incremental deletion**"""

df_inc = replace_and_get_accuracy_flip_lor(shap_values, tokenizer, model, labels, original_probs, device, 'global', 'top',  k_range = np.arange(0, 1.01, 0.05))
df_inc.to_csv(path_final+dataset+'_inc.csv')

df_inc

#Hide FONT warning
import logging
logging.getLogger('matplotlib.font_manager').setLevel(level=logging.CRITICAL)

test='inc'
plot_name=(xai_method_name + ' | ' + model_name + ' - ' + dataset + ' Incremental Deletion')
df=df_inc
line_plot_template(df['acc_selected'],  df['acc_rand'],  'Accuracy', ["Accuracy", "Random Accuracy"], (test+"_plot_acc.png") , plot_name, 'k', df_inc['k'])

line_plot_template(df['aopc_selected'],  df['aopc_rand'],  'AOPC', ["AOPC", "Random AOPC"], (dataset+'_'+test+"_plot_aopc.png"), plot_name, 'k', df_inc['k'])

line_plot_template(df['lor_change_selected'],  df['lor_change_rand'],  'Log-odds', ["LOR", "Random LOR"], (dataset+'_'+test+"_plot_lor.png"), plot_name, 'k', df_inc['k'])

y=df['acc_selected']; z= df['acc_rand']; ylabel= 'Accuracy';  title= plot_name; xlabel= 'k'; x=df['k']
plt.figure(figsize=(10, 5))
plt.plot(x, y, color='blue', marker='o')
plt.plot(x, z, color='red', marker='o')
plt.plot(x, df['flip_selected'], color='green', marker='o')
plt.plot(x, df['flip_rand'], color='yellow', marker='o')
plt.rcParams["font.family"] = "Arial"
plt.title(title)
plt.xlabel("k", fontsize=14)
plt.ylabel("Accuracy & Flip", fontsize=14)
plt.legend(["Accuracy", "Random Accuracy", "Flip", "Random Flip"], fontsize=14)
plt.grid(visible=True, which='both', axis='both', linestyle='--', color='grey')
plt.savefig((path_final+dataset+test+"_plot_flip.png"), dpi=300, bbox_inches='tight')
plt.show()

"""# **incremental addition**"""

df_inc_add= replace_and_get_accuracy_flip_lor(shap_values, tokenizer, model, labels, original_probs, device, 'global', 'bottom',  k_range = np.arange(0, 1.01, 0.05))
df_inc_add.to_csv(path_final+dataset+'_inc_add.csv')

df_inc_add.to_csv(path_final+dataset+'_inc_add.csv')

df_inc_add

test='inc_add'
plot_name=(xai_method_name + ' | ' + model_name + ' - ' + dataset +' Incremental Addition')
df=df_inc_add
line_plot_template(df['acc_selected'],  df['acc_rand'],  'Accuracy', ["Accuracy", "Random Accuracy"], (test+"_plot_acc.png") , plot_name, 'k', df['k'])

line_plot_template(df['lor_change_selected'],  df['lor_change_rand'],  'Log-odds', ["LOR", "Random LOR"], (test+"_plot_lor.png"), plot_name, 'k', df['k'])

line_plot_template(df['aopc_selected'],  df['aopc_rand'],  'AOPC', ["AOPC", "Random AOPC"], (test+"_plot_aopc.png"), plot_name, 'k', df['k'])

y=df['acc_selected']; z= df['acc_rand']; ylabel= 'Accuracy';  title= plot_name; xlabel= 'k'; x=df['k']
plt.figure(figsize=(10, 5))
plt.plot(x, y, color='blue', marker='o')
plt.plot(x, z, color='red', marker='o')
plt.plot(x, df['flip_selected'], color='green', marker='o')
plt.plot(x, df['flip_rand'], color='yellow', marker='o')
plt.rcParams["font.family"] = "Arial"
plt.title(title)
plt.xlabel("k", fontsize=14)
plt.ylabel("Accuracy & Flip", fontsize=14)
plt.legend(["Accuracy", "Random Accuracy", "Flip", "Random Flip"], fontsize=14)
plt.grid(visible=True, which='both', axis='both', linestyle='--', color='grey')
plt.savefig((path_final+dataset+test+"_plot_flip.png"), dpi=300, bbox_inches='tight')
plt.show()

"""# **deletion check**"""

df_del = replace_and_get_accuracy_flip_lor(shap_values, tokenizer, model, labels, original_probs, device, 'local', 'top', k_range = np.arange(0, 15, 1), ratio='on') #np.arange(0, 140, 5))
df_del.to_csv(path_final+dataset+'_del.csv')

df_del

test='del'
plot_name=(xai_method_name + ' | ' + model_name + ' - ' + dataset +' Deletion Check')
df=df_del
line_plot_template(df['acc_selected'],  df['acc_rand'],  'Accuracy', ["Accuracy", "Random Accuracy"], (test+dataset+"_plot_acc.png") , plot_name, 'k', df['k'])

line_plot_template(df['lor_change_selected'],  df['lor_change_rand'],  'Log-odds', ["LOR", "Random LOR"], (test+dataset+"_plot_lor.png"), plot_name, 'k', df['k'])

line_plot_template(df['aopc_selected'],  df['aopc_rand'],  'AOPC', ["AOPC", "Random AOPC"], (test+dataset+"_plot_aopc.png"), plot_name, 'k', df['k'])

y=df['acc_selected']; z= df['acc_rand']; ylabel= 'Accuracy';  title= plot_name; xlabel= 'k'; x=df['k']
plt.figure(figsize=(10, 5))
plt.plot(x, y, color='blue', marker='o')
plt.plot(x, z, color='red', marker='o')
plt.plot(x, df['flip_selected'], color='green', marker='o')
plt.plot(x, df['flip_rand'], color='yellow', marker='o')
plt.rcParams["font.family"] = "Arial"
plt.title(title)
plt.xlabel("k", fontsize=14)
plt.ylabel("Accuracy & Flip", fontsize=14)
plt.legend(["Accuracy", "Random Accuracy", "Flip", "Random Flip"], fontsize=14)
plt.grid(visible=True, which='both', axis='both', linestyle='--', color='grey')
plt.savefig((path_final+dataset+test+"_plot_flip.png"), dpi=300, bbox_inches='tight')
plt.show()

"""# **preservation check**"""

df_pres= replace_and_get_accuracy_flip_lor(shap_values, tokenizer, model, labels, original_probs, device, 'local', 'bottom', k_range = np.arange(0, 15, 1), ratio='on') #np.arange(0, 140, 5))

df_pres.to_csv(path_final+dataset+'_pres.csv')

df_pres

test='pres'
plot_name=(xai_method_name + ' | ' + model_name + ' - ' + dataset +' Preservation Check')
df=df_pres
line_plot_template(df['acc_selected'],  df['acc_rand'],  'Accuracy', ["Accuracy", "Random Accuracy"], (test+dataset+"_plot_acc.png") , plot_name, 'k', df['k'])

line_plot_template(df['lor_change_selected'],  df['lor_change_rand'],  'Log-odds', ["LOR", "Random LOR"], (test+dataset+"_plot_lor.png"), plot_name, 'k', df['k'])

line_plot_template(df['aopc_selected'],  df['aopc_rand'],  'AOPC', ["AOPC", "Random AOPC"], (test+dataset+"_plot_aopc.png"), plot_name, 'k', df['k'])

y=df['acc_selected']; z= df['acc_rand']; ylabel= 'Accuracy';  title= plot_name; xlabel= 'k'; x=df['k']
plt.figure(figsize=(10, 5))
plt.plot(x, y, color='blue', marker='o')
plt.plot(x, z, color='red', marker='o')
plt.plot(x, df['flip_selected'], color='green', marker='o')
plt.plot(x, df['flip_rand'], color='yellow', marker='o')
plt.rcParams["font.family"] = "Arial"
plt.title(title)
plt.xlabel("k", fontsize=14)
plt.ylabel("Accuracy & Flip", fontsize=14)
plt.legend(["Accuracy", "Random Accuracy", "Flip", "Random Flip"], fontsize=14)
plt.grid(visible=True, which='both', axis='both', linestyle='--', color='grey')
plt.savefig((path_final+dataset+test+"_plot_flip.png"), dpi=300, bbox_inches='tight')
plt.show()

"""# **CREATE SUMMARY GRAPHS (run separately)**"""

!pip install transformers
!pip install shap

#from transformers import BertTokenizer, BertForSequenceClassification
from transformers import RobertaTokenizer, RobertaForSequenceClassification

import pickle
import numpy as np
import pandas as pd
import random
from itertools import chain
import torch
import torch.nn.functional as F
import itertools
import heapq

import matplotlib.pyplot as plt
!apt-get install -y fonts-arial-regular
plt.rcParams["font.family"] = "Arial"

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device("cpu")

#set XAI method name

#xai_method_name = "SHAP"
#xai_method_name = "LIME"
xai_method_name = "VANILLA_GRADIENTS"

"""# load data"""

path_data_created='/content/drive/MyDrive/benchmark/final_files/'
datasets=['ISOT', 'EUvsDISINFO', 'EU_ENR_MIX4', 'EU_EMNAD_MIX3']
path_data_created_sets=['eval1/ISOT/'+xai_method_name+'/', 'eval1/EUvsDISINFO/'+xai_method_name+'/', 'eval1/EU_ENR_MIX4/'+xai_method_name+'/', 'eval1/EU_EMNAD_MIX3/'+xai_method_name+'/']
#SHAP / LIME / VANILLA GRADIENTS - three summary graphs in total featuring all four datasets and two models (one per dataset)

df_del = {}
df_pres = {}
df_inc = {}
df_inc_add = {}

for i in range(len(datasets)):
    dataset = datasets[i]
    path_data_created_set = path_data_created_sets[i]
    df_del[dataset] = pd.read_csv(path_data_created+path_data_created_set+dataset+'_del.csv')
    df_pres[dataset] = pd.read_csv(path_data_created+path_data_created_set+dataset+'_pres.csv')
    df_inc[dataset] = pd.read_csv(path_data_created+path_data_created_set+dataset+'_inc.csv')
    df_inc_add[dataset] = pd.read_csv(path_data_created+path_data_created_set+dataset+'_inc_add.csv')

for d in datasets:
  df_inc[d].to_latex((path_data_created+'/eval1/'+d+'/'+d+'_df_inc.tex')) #TEST GRAPHS

#df_inc_add['ISOT']

"""# definitions"""

#linear interpolation (for DELETION and PRESERVATION CHECK)
#https://stackoverflow.com/questions/66934748/how-to-stretch-an-array-to-a-new-length-while-keeping-same-value-distribution

def interp1d(array: np.ndarray, new_len: int) -> np.ndarray:
    la = len(array)
    return np.interp(np.linspace(0, la - 1, num=new_len), np.arange(la), array)

def line_plot_template(data, ylabel, legend, filename, title, xlabel, x, x_ax_range=None, y_ax_range=None, colors=None, markers=None, interpol=None):
    plt.figure(figsize=(11.69, 8.27))

    if interpol=='on':
      d_max_len=0
      for d in data:
        if d_max_len < len(d):
          d_max_len=len(d)
      #print("max len:")
      #print(d_max_len)
      cnt=0
      while cnt < len(data):
        if(len(data[cnt]) == d_max_len):
          #print("No interpolation due to same length.")
          cnt+=1
        else:
          #print("Start interpolations, original len:")
          #print(len(data[cnt]))
          #print(data[cnt][:10])
          data[cnt] = interp1d(data[cnt], new_len=d_max_len)
          #print("After interpolation len:")
          #print(len(data[cnt]))
          cnt+=1
          #print("Samples:")
          #print(data[cnt][:10])
      #print(len(data)) #8
      #print(data[0])

    if colors is None:
        # colors: red, blue, green,  Burgundy, Navy blue, Olive Drab, gold, Dark khaki ,'#ffd700', '#cfb53b'
        # colors:
        #colors = ['#ff0051', '#008bfb', '#2cb32c', '#ff0051', '#008bfb', '#2cb32c', '#8b0000', '#00008b']#, '#134d13']
        colors = ['#f91010', '#0456f8', '#00e507', '#a504f1', '#f78b8b', '#6194f8', '#70de74', '#c573ec']
    if markers is None:
        markers = ['o','*','x','_','+','v','p','D']
    if len(data) != len(legend):
        raise ValueError("The number of lines to be plotted must match the length of the legend list.")
    for i, d in enumerate(data):
      if i >= len(data)/2:
        plt.plot(x, d, color=colors[i], marker=markers[i], markersize=10, linestyle = '--', alpha=0.35) #marker='o'
      else:
        plt.plot(x, d, color=colors[i], marker=markers[i], markersize=10) #marker='o'
    plt.rcParams["font.family"] = "Arial"
    plt.xlim(x_ax_range)
    plt.ylim(y_ax_range)
    plt.title(title, fontsize=22)
    plt.xlabel(xlabel, fontsize=22)
    plt.ylabel(ylabel, fontsize=22)
    #plt.legend(legend, fontsize=14)
    plt.legend(legend, fontsize=22, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0)
    plt.grid(visible=True, which='both', axis='both', linestyle='--', color='gainsboro')
    plt.rcParams['xtick.labelsize'] = 20
    plt.rcParams['ytick.labelsize'] = 20
    plt.savefig((path_data_created+'eval1/'+xai_method_name+'_'+filename), dpi=300, bbox_inches='tight')
    plt.show()

"""# **GRAPH CREATION**"""

df_del['ISOT'].columns
col=['Unnamed: 0', 'k', 'acc_selected', 'flip_selected',
       'lor_change_selected', 'aopc_selected', 'acc_rand', 'flip_rand',
       'lor_change_rand', 'aopc_rand']

#[print(ds, col)for col in cols for ds in datasets ]

#INC DEL
#[print(ds, col)for col in cols for ds in datasets ]

cols=['lor_change_selected', 'lor_change_rand']
data = [df_inc[ds][col] for col in cols for ds in datasets ]
x = df_inc['ISOT']['k']
#legend = ['ISOT', 'EUvsDISINFO', 'EU_ENR_MIX4', 'EU_EMNAD_MIX3', 'ISOT random', 'EUvsDISINFO random', 'EU_ENR_MIX4 random', 'EU_EMNAD_MIX3 random']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Log-odds rate'
filename = 'lor_inc.png'
title = xai_method_name.replace("_", " ")+' | '+'Log-odds rate for Incremental Deletion'
xlabel = 'k'
line_plot_template(data, ylabel, legend, filename, title, xlabel, x,)

cols=['aopc_selected', 'aopc_rand']
data = [df_inc[ds][col] for col in cols for ds in datasets ]
x = df_inc['ISOT']['k']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'AOPC'
filename = 'aopc_inc.png'
title = xai_method_name.replace("_", " ")+' | '+'AOPC for Incremental Deletion'
xlabel = 'k'
line_plot_template(data, ylabel, legend, filename, title, xlabel, x,)

#INC DEL
cols=['acc_selected', 'acc_rand']
data = [df_inc[ds][col] for col in cols for ds in datasets ]
x = df_inc['ISOT']['k']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Accuracy'
filename_del = 'acc_inc_del.png'
title = xai_method_name.replace("_", " ")+' | '+'Accuracy for Incremental Deletion'
xlabel = 'k'
line_plot_template(data, ylabel, legend, filename_del, title, xlabel, x,)

#INC ADD
cols=['lor_change_selected', 'lor_change_rand']
data = [df_inc_add[ds][col] for col in cols for ds in datasets ]
x = df_inc_add['ISOT']['k']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Log-odds rate'
filename = 'lor_inc_add.png'
title = xai_method_name.replace("_", " ")+' | '+'Log-odds rate for Incremental Addition'
xlabel = 'k'
line_plot_template(data, ylabel, legend, filename, title, xlabel, x,)

cols=['aopc_selected', 'aopc_rand']
data = [df_inc_add[ds][col] for col in cols for ds in datasets ]
x = df_inc_add['ISOT']['k']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'AOPC'
filename_del = 'aopc_inc_add.png'
title = xai_method_name.replace("_", " ")+' | '+'AOPC for Incremental Addition'
xlabel = 'k'
line_plot_template(data, ylabel, legend, filename_del, title, xlabel, x,)

cols=['acc_selected', 'acc_rand']
data = [df_inc_add[ds][col] for col in cols for ds in datasets ]
x = df_inc_add['ISOT']['k']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Accuracy'
filename_del = 'acc_inc_add.png'
title = xai_method_name.replace("_", " ")+' | '+'Accuracy for Incremental Addition'
xlabel = 'k'
line_plot_template(data, ylabel, legend, filename_del, title, xlabel, x,)

'''

#ADJUST ratio lengths to match (preservation check summary graphs) (NOT needed when using interpolation)

#get longest df
m = max(len(df_del['ISOT']),len(df_del['EUvsDISINFO']),len(df_del['EU_ENR_MIX4']),len(df_del['EU_EMNAD_MIX3']))
print(m)

#fill empty rows with zeros so the plot can be made (if no interpolation is used):
for ds in datasets:
  while len(df_del[ds]) < m:
    df_del[ds].loc[len(df_del[ds])] = pd.Series(dtype='float64')

#print(df_del['ISOT']['k']) #0-78
#print(df_del['EUvsDISINFO']['k']) #0-16
#print(df_del['EU_ENR_MIX4']['k']) #0-58
#print(df_del['EU_EMNAD_MIX3']['k']) #0-55
#print(df_del['ISOT'])
#print(df_del['EUvsDISINFO'])
'''

#DELETION CHECK

cols=['lor_change_selected', 'lor_change_rand']
data = [df_del[ds][col]  for col in cols for ds in datasets ]
x = df_del['ISOT']['k']
#x = np.arange(0,m) #alternative (above works too because ISOT usually the longest avg)
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Log-odds rate'
filename_del = 'lor_change_selected_del.png'
title = xai_method_name.replace("_", " ")+' | '+'Log-odds rate for Deletion Check'
xlabel = 'k (interpolated)'
line_plot_template(data, ylabel, legend, filename_del, title, xlabel, x, interpol='on')

cols=['aopc_selected', 'aopc_rand']
data = [df_del[ds][col] for col in cols for ds in datasets ]
x = df_del['ISOT']['k']
#x = np.arange(0,m) (above works too because ISOT usually the longest avg)
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Log-odds rate'
filename_del = 'aopc_change_selected_del.png'
title = xai_method_name.replace("_", " ")+' | '+'AOPC rate for Deletion Check'
xlabel = 'k (interpolated)'
line_plot_template(data, ylabel, legend, filename_del, title, xlabel, x, interpol='on')

#DELETION CHECK
cols=['acc_selected', 'acc_rand']
data = [df_del[ds][col] for col in cols for ds in datasets ]
x = df_del['ISOT']['k']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Acc-reduced'
filename = 'acc_del.png'
title = xai_method_name.replace("_", " ")+' | '+'Acc-reduced for Deletion Check'
xlabel = 'k (interpolated)'
line_plot_template(data, ylabel, legend, filename, title, xlabel, x, interpol='on')

cols=['flip_selected', 'flip_rand']
data = [df_del[ds][col] for col in cols for ds in datasets ]
x = df_del['ISOT']['k']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Label flip'
filename = 'flip_del.png'
title = xai_method_name.replace("_", " ")+' | '+'Label flip for Deletion Check'
xlabel = 'k (interpolated)'
line_plot_template(data, ylabel, legend, filename, title, xlabel, x, interpol='on')

'''
#ADJUST ratio lengths to match (preservation check summary graphs) (NOT needed when using interpolation)

#get longest df
m = max(len(df_pres['ISOT']),len(df_pres['EUvsDISINFO']),len(df_pres['EU_ENR_MIX4']),len(df_pres['EU_EMNAD_MIX3']))
#print(m) #=79

for ds in datasets:
  while len(df_pres[ds]) < m:
    df_pres[ds].loc[len(df_pres[ds])] = pd.Series(dtype='float64')
'''

#PRES CHECK
cols=['acc_selected', 'acc_rand']
data = [df_pres[ds][col] for col in cols for ds in datasets ]
x = df_pres['ISOT']['k']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Acc-reduced'
filename = 'acc_pres.png'
title = xai_method_name.replace("_", " ")+' | '+'Acc-reduced for Preservation Check'
xlabel = 'k (interpolated)'
line_plot_template(data, ylabel, legend, filename, title, xlabel, x, interpol='on')

cols=['flip_selected', 'flip_rand']
data = [df_pres[ds][col] for col in cols for ds in datasets ]
x = df_pres['ISOT']['k']
legend = ['ISOT', 'EUvsDISINFO', 'ENR', 'EMNAD', 'ISOT random', 'EUvsDISINFO random', 'ENR random', 'EMNAD random']
ylabel = 'Label flip'
filename = 'flip_pres.png'
title = xai_method_name.replace("_", " ")+' | '+'Label flip for Preservation Check'
xlabel = 'k (interpolated)'
line_plot_template(data, ylabel, legend, filename, title, xlabel, x, interpol='on')