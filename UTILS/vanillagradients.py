# -*- coding: utf-8 -*-

#RobertaForSequenceClassification

import tensorflow as tf
import torch as pt
import numpy as np

#adapted, based on tf code by https://victordibia.com/blog/explain-bert-classification/

def get_gradients(text, model, tokenizer):

  def get_correct_span_mask(correct_index, token_size):
    span_mask = np.zeros((1, token_size))
    span_mask[0, correct_index] = 1
    span_mask = pt.tensor(span_mask, dtype=pt.int32)
    return span_mask

  embedding_matrix = model.to('cuda').roberta.embeddings.word_embeddings.weight.data
  #print(embedding_matrix)
  embedding_matrix = embedding_matrix.cpu().detach().numpy()
  embedding_matrix = pt.from_numpy(embedding_matrix).to('cuda')
  #print(embedding_matrix)
  encoded_tokens = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
  #print(encoded_tokens)
  token_ids = list(encoded_tokens["input_ids"].numpy()[0])
  #print(token_ids)
  vocab_size = embedding_matrix.shape[0] #get_shape()[0]
  #print(vocab_size)

  # convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
  token_ids_tensor = pt.tensor([token_ids], dtype=pt.int32)
  #print(token_ids_tensor)
  token_ids_tensor_one_hot = pt.nn.functional.one_hot(token_ids_tensor.long(), vocab_size)
  #print(token_ids_tensor_one_hot)

  # (i) watch input variable
  token_ids_tensor_one_hot = token_ids_tensor_one_hot.clone().detach().float().requires_grad_(True)

  # multiply input model embedding matrix; allows us do backprop wrt one hot input
  #print(token_ids_tensor_one_hot)
  #print(embedding_matrix)
  inputs_embeds = pt.matmul(token_ids_tensor_one_hot.to(pt.float32).to('cuda'), embedding_matrix)

  #print(inputs_embeds)
  #print(encoded_tokens["attention_mask"])

  # (ii) get prediction
  pred_scores = model(inputs_embeds=inputs_embeds, attention_mask=encoded_tokens["attention_mask"])[0]#.logits
  max_class = pt.argmax(pred_scores, axis=1).cpu().numpy()[0]

  # get mask for predicted score class
  score_mask = get_correct_span_mask(max_class, pred_scores.shape[1])

  # zero out all predictions outside of the correct  prediction class; we want to get gradients wrt to just this class
  predict_correct_class = pt.sum(pred_scores.to('cuda') * score_mask.to('cuda') ) #tf.reduce_sum()
  predict_correct_class.backward()

  # (iii) get gradient of input with respect to prediction class
  grad_res = token_ids_tensor_one_hot.grad
  #print(grad_res)
  gradient_non_normalized = pt.linalg.norm(grad_res, dim=2)
  #print(gradient_non_normalized)

  # (iv) normalize gradient scores and return them as "explanations"
  gradient_tensor = (
      gradient_non_normalized /
      pt.max(gradient_non_normalized) #tf.reduce_max()
  )

  gradients = gradient_tensor[0].numpy().tolist()
  #print(gradients)
  token_words = tokenizer.convert_ids_to_tokens(token_ids)

  prediction_label= "TRUE" if max_class == 1 else "FALSE"
  
  return gradients, token_words , prediction_label