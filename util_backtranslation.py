"""Utility classes and methods.

Author:
    Chris Chute (chute@stanford.edu)
"""
from ast import Index
import logging
import os
import queue
import re
import shutil
import string as stringutils
import torch
import torch.nn.functional as F
import torch.utils.data as data
import tqdm
import numpy as np
import ujson as json
import random
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from joblib import Parallel, delayed
import multiprocessing

class SQuAD(data.Dataset):
    """Stanford Question Answering Dataset (SQuAD).

    Each item in the dataset is a tuple with the following entries (in order):
        - context_idxs: Indices of the words in the context.
            Shape (context_len,).
        - context_char_idxs: Indices of the characters in the context.
            Shape (context_len, max_word_len).
        - question_idxs: Indices of the words in the question.
            Shape (question_len,).
        - question_char_idxs: Indices of the characters in the question.
            Shape (question_len, max_word_len).
        - y1: Index of word in the context where the answer begins.
            -1 if no answer.
        - y2: Index of word in the context where the answer ends.
            -1 if no answer.
        - id: ID of the example.

    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        use_v2 (bool): Whether to use SQuAD 2.0 questions. Otherwise only use SQuAD 1.1.
    """
    def __init__(self, data_path, use_v2=True):
        super(SQuAD, self).__init__()

        dataset = np.load(data_path)
        self.context_idxs = torch.from_numpy(dataset['context_idxs']).long()
        self.context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
        self.question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
        self.question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
        self.y1s = torch.from_numpy(dataset['y1s']).long()
        self.y2s = torch.from_numpy(dataset['y2s']).long()

        if use_v2:
            # SQuAD 2.0: Use index 0 for no-answer token (token 1 = OOV)
            batch_size, c_len, w_len = self.context_char_idxs.size()
            ones = torch.ones((batch_size, 1), dtype=torch.int64)
            self.context_idxs = torch.cat((ones, self.context_idxs), dim=1)
            self.question_idxs = torch.cat((ones, self.question_idxs), dim=1)

            ones = torch.ones((batch_size, 1, w_len), dtype=torch.int64)
            self.context_char_idxs = torch.cat((ones, self.context_char_idxs), dim=1)
            self.question_char_idxs = torch.cat((ones, self.question_char_idxs), dim=1)

            self.y1s += 1
            self.y2s += 1

        # SQuAD 1.1: Ignore no-answer examples
        self.ids = torch.from_numpy(dataset['ids']).long()
        self.valid_idxs = [idx for idx in range(len(self.ids))
                           if use_v2 or self.y1s[idx].item() >= 0]

    def __getitem__(self, idx):
        idx = self.valid_idxs[idx]
        example = (self.context_idxs[idx],
                   self.context_char_idxs[idx],
                   self.question_idxs[idx],
                   self.question_char_idxs[idx],
                   self.y1s[idx],
                   self.y2s[idx],
                   self.ids[idx])
        return example

    def __len__(self):
        return len(self.valid_idxs)
        
def span_corrupt(data_path, output_path):
    """Stanford Question Answering Dataset (SQuAD).

    Each item in the dataset is a tuple with the following entries (in order):
        - context_idxs: Indices of the words in the context.
            Shape (context_len,).
        - context_char_idxs: Indices of the characters in the context.
            Shape (context_len, max_word_len).
        - question_idxs: Indices of the words in the question.
            Shape (question_len,).
        - question_char_idxs: Indices of the characters in the question.
            Shape (question_len, max_word_len).
        - y1: Index of word in the context where the answer begins.
            -1 if no answer.
        - y2: Index of word in the context where the answer ends.
            -1 if no answer.
        - id: ID of the example.

    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        use_v2 (bool): Whether to use SQuAD 2.0 questions. Otherwise only use SQuAD 1.1.
    """
    print("Starting to create the span corruption dataset.")

    dataset = np.load(data_path)
    context_idxs = torch.from_numpy(dataset['context_idxs']).long()
    context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
    question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
    question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
    y1s = torch.from_numpy(dataset['y1s']).long()
    y2s = torch.from_numpy(dataset['y2s']).long()

    ids = torch.from_numpy(dataset['ids']).long()
    last_id = ids[-1].item()

    add_context_idxs = []
    add_context_chars_idxs = []
    add_question_idxs = []
    add_question_chars_idxs = []
    add_y1s = []
    add_y2s = []
    add_ids = []

    for idx in range(len(ids)): 
        if idx % 1000 == 0:
            print("Processing " + str(idx) + "th training entry.")
        new_y1 = y1s[idx]
        new_y2 = y2s[idx]

        new_context_idxs = context_idxs[idx]
        new_context_chars_idxs = context_char_idxs[idx]

        trunc_len = random.randint(4, int(len(new_context_idxs)*7/8))
        trunc_doc = new_context_idxs[:trunc_len]
        masked_content_len = int(trunc_len // 4) # we're always masking a quarter of the truncated content

        end_prefix = ((trunc_len - masked_content_len) // 2) - 1
        prefix = trunc_doc[:end_prefix]
        masked_content = trunc_doc[end_prefix:end_prefix + masked_content_len]
        masked_content = torch.tensor([1 for elem in masked_content]) # replace them with NULL tokens
        suffix = trunc_doc[end_prefix + masked_content_len:]
        add_new_context_idxs = torch.cat((prefix, masked_content, suffix, new_context_idxs[trunc_len:]), 0) 

        # handle self.context_char_idxs[idx]
        prefix_char = new_context_chars_idxs[:end_prefix] 
        masked_content_char = new_context_chars_idxs[end_prefix:end_prefix + masked_content_len]
        masked_content_char = torch.tensor([[1] * masked_content_char.size()[1] for i in range(masked_content_char.size()[0])])
        suffix_char = new_context_chars_idxs[end_prefix + masked_content_len:]
        add_new_context_chars_idxs = torch.cat((prefix_char, masked_content_char, suffix_char), 0)

        # If the original answer is not preserved, then make sure that there is no answer.  
        # Corrupting only part of the answer means that it won't always be a valid answer. 
        if new_context_idxs[y1s[idx].item()] != add_new_context_idxs[y1s[idx].item()] \
            or new_context_idxs[y2s[idx].item()] != add_new_context_idxs[y2s[idx].item()]:
            new_y1, new_y2 = torch.tensor(-1), torch.tensor(-1)

        if idx % 1000 == 0:
            print("Adding tensors")
        # apply the unsqueeze to properly concatenate the things
        add_context_idxs.append(add_new_context_idxs.tolist())
        add_context_chars_idxs.append(add_new_context_chars_idxs.tolist())
        add_question_idxs.append(question_idxs[idx].tolist())
        add_question_chars_idxs.append(question_char_idxs[idx].tolist())
        add_y1s.append(new_y1.tolist())
        add_y2s.append(new_y2.tolist())
        add_ids.append(ids[idx].item() + last_id)
        
    print("finished creating all tensors")

    context_idxs = torch.cat((context_idxs, torch.tensor(add_context_idxs)), 0)
    context_char_idxs = torch.cat((context_char_idxs, torch.tensor(add_context_chars_idxs)), 0)

    question_idxs = torch.cat((question_idxs, torch.tensor(add_question_idxs)), 0)
    question_char_idxs = torch.cat((question_char_idxs, torch.tensor(add_question_chars_idxs)), 0)  

    y1s = torch.cat((y1s, torch.tensor(add_y1s)), 0)
    y2s = torch.cat((y2s, torch.tensor(add_y2s)), 0)

    ids = torch.cat((ids, torch.tensor(add_ids)), 0)  

    # save into a new file
    print("Finished processing all training examples. Saving into file.")
    np.savez_compressed(output_path, 
                        context_idxs=context_idxs, 
                        context_char_idxs=context_char_idxs, 
                        ques_idxs=question_idxs, 
                        ques_char_idxs=question_char_idxs, 
                        y1s=y1s, 
                        y2s=y2s, 
                        ids=ids)
    print("Finished saving data into file.")

def convert_to_string(indices, dictionary):
    words = [dictionary[elem.item()] for elem in indices]
    words = [elem if elem != '--NULL--' else '' for elem in words]
    return " ".join(words)

# for strings with words
def convert_to_indices(string, dictionary, orig_word_idxs):
    string = re.sub(r'[^\w\s]', '', string)
    string = string.split()
    idxs = []
    for elem in string:
        if elem in dictionary:
            idxs.append(dictionary[elem])
    for i in range(len(idxs), len(orig_word_idxs)):
        idxs.append(0)
    return torch.tensor(idxs)

# for getting the character indices
def convert_to_char_indices(string, orig_char_idxs, dictionary, word_dictionary):
    string = re.sub(r'[^\w\s]', '', string)
    string = string.split()
    # chars = [list(word) if (word != "--NULL--" or "" or " ") else [] for word in string]
    chars = []
    nonsense_words = 0
    for word in string:
        if word in word_dictionary and (word != "--NULL--"):
            chars.append(list(word))
        else:
            nonsense_words += 1
    for i in range(nonsense_words):
        chars.append([])
    new_char_idxs = orig_char_idxs.clone().detach()
    for i in range(len(chars)):
        word = chars[i]
        for j in range(len(word)):
            char = word[j]
            new_char_idxs[i][j] = dictionary[char]
        for j in range(len(word), new_char_idxs.shape[1]):
            new_char_idxs[i][j] = 0
    return torch.tensor(new_char_idxs)

# increase batch size of translations
def translate(input, model, tokenizer):
    batch = tokenizer(input, return_tensors="pt", padding=True)
    gen = model.generate(**batch)
    output = tokenizer.batch_decode(gen, skip_special_tokens=True)
    return output

def bigrams(string):
    bigrams_list = []
    if len(string) == 1:
        string += ' '
    for i in range(len(string) - 1):
        new_str = string[i] + string[i+1]
        bigrams_list.append(new_str)
    return bigrams_list

def jaccard_similarity(list1, list2):
    set1 = set(list1)
    set2 = set(list2)
    
    numerator = len(set1.intersection(set2))
    denom = len(set1.union(set2))
    
    return numerator / denom

def find_most_similar(answer, context, bigrams, jaccard_similarity):
    similar = {}
    answer_bigrams = bigrams(answer)
    for i in range(len(context)):
        word = context[i]
        word_bigrams = bigrams(word)
        similarity = jaccard_similarity(word_bigrams, answer_bigrams)
        if similarity >= 0.5:
            similar[i] = similarity
    return similar

def find_best_answer(start_candidates, end_candidates, trans_context_words, orig_answer_words):
    best_answer = []
    best_answer_score = 0
    for end_idx in end_candidates:
        for start_idx in start_candidates:
            if start_idx <= end_idx:
                score = jaccard_similarity(trans_context_words[start_idx:end_idx + 1], orig_answer_words)
                if score > best_answer_score:
                    best_answer = [start_idx, end_idx]
                    best_answer_score = score
    if best_answer_score < 0.25:
        return torch.tensor([-1, -1])
    return torch.tensor(best_answer)

def back_translation(data_path, output_path):
    """Stanford Question Answering Dataset (SQuAD).

    Each item in the dataset is a tuple with the following entries (in order):
        - context_idxs: Indices of the words in the context.
            Shape (context_len,).
        - context_char_idxs: Indices of the characters in the context.
            Shape (context_len, max_word_len).
        - question_idxs: Indices of the words in the question.
            Shape (question_len,).
        - question_char_idxs: Indices of the characters in the question.
            Shape (question_len, max_word_len).
        - y1: Index of word in the context where the answer begins.
            -1 if no answer.
        - y2: Index of word in the context where the answer ends.
            -1 if no answer.
        - id: ID of the example.

    Args:
        data_path (str): Path to .npz file containing pre-processed dataset.
        use_v2 (bool): Whether to use SQuAD 2.0 questions. Otherwise only use SQuAD 1.1.
    """

    print("Starting to create the backtranslation dataset.")
    
    dataset = np.load(data_path)
    context_idxs = torch.from_numpy(dataset['context_idxs']).long()
    context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
    question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
    question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
    y1s = torch.from_numpy(dataset['y1s']).long()
    y2s = torch.from_numpy(dataset['y2s']).long()

    ids = torch.from_numpy(dataset['ids']).long()
    last_id = ids[-1].item()

    en_de_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
    en_de_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

    de_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
    de_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

    word2idx = json.loads(open('./data/word2idx.json', 'r').read())
    char2idx = json.loads(open('./data/char2idx.json', 'r').read())

    idx2word = dict((v,k) for k,v in word2idx.items())
    idx2char = dict((v,k) for k,v in char2idx.items())

    add_context_idxs = []
    add_context_chars_idxs = []
    add_question_idxs = []
    add_question_chars_idxs = []
    add_y1s = []
    add_y2s = []
    add_ids = []

    batch_size = 10
    num_cores = multiprocessing.cpu_count()

    # try to parallize the batches all at once. 
    for idx in range(0, len(ids), batch_size): 
        print("Processed " + str(idx) + " training entries.")
        new_y1s = y1s[idx:idx + batch_size]
        new_y2s = y2s[idx:idx + batch_size]

        new_context_idxs_list = context_idxs[idx:idx + batch_size]
        new_context_chars_idxs_list = context_char_idxs[idx:idx + batch_size]

        new_question_idxs_list = question_idxs[idx:idx + batch_size]
        new_question_char_idxs_list = question_char_idxs[idx:idx + batch_size]
        
        question_strings = Parallel(n_jobs=num_cores)(delayed(convert_to_string)(i, idx2word) for i in new_question_idxs_list)
        
        trans_question_strings_list = translate(translate(question_strings, en_de_model, en_de_tokenizer), de_en_model, de_en_tokenizer)

        for j in range(len(new_y1s)):
            new_context_idxs = new_context_idxs_list[j]
            new_context_chars_idxs = new_context_chars_idxs_list[j]
            
            new_question_idxs = new_question_idxs_list[j]
            new_question_char_idxs = new_question_char_idxs_list[j]

            new_y1 = new_y1s[j]
            new_y2 = new_y2s[j]

            trans_question_strings = trans_question_strings_list[j]

            add_new_question_idxs = convert_to_indices(trans_question_strings, word2idx, new_question_idxs)       
            add_new_question_char_idxs = convert_to_char_indices(trans_question_strings, new_question_char_idxs, char2idx, word2idx)
            

            add_context_idxs.append(new_context_idxs.tolist())
            add_context_chars_idxs.append(new_context_chars_idxs.tolist())
            add_question_idxs.append(add_new_question_idxs.tolist())
            add_question_chars_idxs.append(add_new_question_char_idxs.tolist())
            add_y1s.append(new_y1.tolist())
            add_y2s.append(new_y2.tolist())
            add_ids.append(ids[idx].item() + last_id)
        print("finished batch")

        
    print("finished creating all tensors")

    context_idxs = torch.cat((context_idxs, torch.tensor(add_context_idxs)), 0)
    context_char_idxs = torch.cat((context_char_idxs, torch.tensor(add_context_chars_idxs)), 0)

    question_idxs = torch.cat((question_idxs, torch.tensor(add_question_idxs)), 0)
    question_char_idxs = torch.cat((question_char_idxs, torch.tensor(add_question_chars_idxs)), 0)  

    y1s = torch.cat((y1s, torch.tensor(add_y1s)), 0)
    y2s = torch.cat((y2s, torch.tensor(add_y2s)), 0)

    ids = torch.cat((ids, torch.tensor(add_ids)), 0)  
    # save into a new file
    print("Finished processing all training examples. Saving into file.")
    np.savez_compressed(output_path, 
                        context_idxs=context_idxs.numpy(), 
                        context_char_idxs=context_char_idxs.numpy(), 
                        ques_idxs=question_idxs.numpy(), 
                        ques_char_idxs=question_char_idxs.numpy(), 
                        y1s=y1s.numpy(), 
                        y2s=y2s.numpy(), 
                        ids=ids.numpy())

    print("Finished saving data into file.")

# def back_translation2(data_path, output_path):
#     """Stanford Question Answering Dataset (SQuAD).

#     Each item in the dataset is a tuple with the following entries (in order):
#         - context_idxs: Indices of the words in the context.
#             Shape (context_len,).
#         - context_char_idxs: Indices of the characters in the context.
#             Shape (context_len, max_word_len).
#         - question_idxs: Indices of the words in the question.
#             Shape (question_len,).
#         - question_char_idxs: Indices of the characters in the question.
#             Shape (question_len, max_word_len).
#         - y1: Index of word in the context where the answer begins.
#             -1 if no answer.
#         - y2: Index of word in the context where the answer ends.
#             -1 if no answer.
#         - id: ID of the example.

#     Args:
#         data_path (str): Path to .npz file containing pre-processed dataset.
#         use_v2 (bool): Whether to use SQuAD 2.0 questions. Otherwise only use SQuAD 1.1.
#     """

#     print("Starting to create the backtranslation dataset.")
    
#     dataset = np.load(data_path)
#     context_idxs = torch.from_numpy(dataset['context_idxs']).long()
#     context_char_idxs = torch.from_numpy(dataset['context_char_idxs']).long()
#     question_idxs = torch.from_numpy(dataset['ques_idxs']).long()
#     question_char_idxs = torch.from_numpy(dataset['ques_char_idxs']).long()
#     y1s = torch.from_numpy(dataset['y1s']).long()
#     y2s = torch.from_numpy(dataset['y2s']).long()

#     ids = torch.from_numpy(dataset['ids']).long()
#     last_id = ids[-1].item()

#     en_de_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
#     en_de_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de")

#     de_en_tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-de-en")
#     de_en_model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-de-en")

#     word2idx = json.loads(open('./data/word2idx.json', 'r').read())
#     char2idx = json.loads(open('./data/char2idx.json', 'r').read())

#     idx2word = dict((v,k) for k,v in word2idx.items())
#     idx2char = dict((v,k) for k,v in char2idx.items())

#     add_context_idxs = []
#     add_context_chars_idxs = []
#     add_question_idxs = []
#     add_question_chars_idxs = []
#     add_y1s = []
#     add_y2s = []
#     add_ids = []

#     batch_size = 10
#     num_cores = multiprocessing.cpu_count()

#     for idx in range(0, len(ids), batch_size): 
#         print("Processed " + str(idx) + " training entries.")
#         new_y1s = y1s[idx:idx + batch_size]
#         new_y2s = y2s[idx:idx + batch_size]

#         new_context_idxs_list = context_idxs[idx:idx + batch_size]
#         new_context_chars_idxs_list = context_char_idxs[idx:idx + batch_size]

#         new_question_idxs_list = question_idxs[idx:idx + batch_size]
#         new_question_char_idxs_list = question_char_idxs[idx:idx + batch_size]
        
#         # retrieve the sentences to use for translation
#         # context_strings = Parallel(n_jobs=num_cores)(delayed(convert_to_string)(i, idx2word) for i in new_context_idxs_list) 
#         question_strings = Parallel(n_jobs=num_cores)(delayed(convert_to_string)(i, idx2word) for i in new_question_idxs_list)
        
#         # convert_to_string(new_context_idxs, idx2word)
#         # question_string = convert_to_string(new_question_idxs, idx2word)

#         # consider translating only the question to speed up 
#         # consider word level substitution
#         # trans_context_strings_list = translate(translate(context_strings, en_de_model, en_de_tokenizer), de_en_model, de_en_tokenizer)
#         trans_question_strings_list = translate(translate(question_strings, en_de_model, en_de_tokenizer), de_en_model, de_en_tokenizer)

#         # find the new answer
#         # first, turn the start word of the answer and the end word of the answer into a 2gram by character level. 
#         # do the 2-gram character of each word for all of the tokens in the new context. 
#         # calculate the most likely candidates for start and end of the answer
#         # calculate the  Jaccard similarity between the sets of character-level 2-grams
#         # in the original answer token and new sentence token -> highest one is the new score
#         for j in range(len(new_y1s)):
#             new_context_idxs = new_context_idxs_list[j]
#             new_context_chars_idxs = new_context_chars_idxs_list[j]
            
#             new_question_idxs = new_question_idxs_list[j]
#             new_question_char_idxs = new_question_char_idxs_list[j]

#             new_y1 = new_y1s[j]
#             new_y2 = new_y2s[j]

#             # trans_context_strings = trans_context_strings_list[j]
#             trans_question_strings = trans_question_strings_list[j]
#             # trans_context_words = trans_context_strings.split()

#             # answer_start_word = idx2word[new_context_idxs[new_y1.item()].item()]
#             # answer_end_word = idx2word[new_context_idxs[new_y2.item()].item()]

#             # answer_idxs = [elem for elem in range(new_y1.item(), new_y2.item() + 1)]
#             # answer_idxs = [new_context_idxs[elem] for elem in answer_idxs]
#             # answer_string = convert_to_string(answer_idxs, idx2word)
        
#             # start_candidates = find_most_similar(answer_start_word, trans_context_words, bigrams, jaccard_similarity)
#             # end_candidates = find_most_similar(answer_end_word, trans_context_words, bigrams, jaccard_similarity)
            
#             # new_y1, new_y2 = find_best_answer(start_candidates, end_candidates, trans_context_words, answer_string.split())
            
#             # apply the unsqueeze to properly concatenate the things
#             # randomly choose original or translated context queries
#             # add_new_context_idxs = convert_to_indices(trans_context_strings, word2idx, context_idxs[idx]) 
#             # add_new_context_chars_idxs = convert_to_char_indices(trans_context_strings, new_context_chars_idxs, char2idx, word2idx)

#             # use_orig_context = np.random.binomial(1, 0.5, 1)[0]
#             # if use_orig_context: # flip a coin
#             #     add_new_context_idxs = new_context_idxs
#             #     add_new_context_chars_idxs = new_context_chars_idxs
#             # context_idxs = torch.cat((context_idxs, torch.unsqueeze(add_new_context_idxs, dim=0)), 0)
#             # context_char_idxs = torch.cat((context_char_idxs, torch.unsqueeze(add_new_context_chars_idxs, dim=0)), 0)

#             add_new_question_idxs = convert_to_indices(trans_question_strings, word2idx, question_idxs[idx])       
#             add_new_question_char_idxs = convert_to_char_indices(trans_question_strings, new_question_char_idxs, char2idx, word2idx)
            
#             # add_new_question_idxs = new_question_idxs
#             # add_new_question_char_idxs = new_question_char_idxs
#             # question_idxs = torch.cat((question_idxs, torch.unsqueeze(add_new_question_idxs, dim=0)), 0)
#             # question_char_idxs = torch.cat((question_char_idxs, torch.unsqueeze(add_new_question_char_idxs, dim=0)), 0)  

#             # fix it so that the y1s and y2s correspond to the correct context
#             # if use_orig_context:
#             #     new_y1 = y1s[idx]
#             #     new_y2 = y2s[idx]

#             add_context_idxs.append(new_context_idxs.tolist())
#             add_context_chars_idxs.append(new_context_chars_idxs.tolist())
#             add_question_idxs.append(add_new_question_idxs.tolist())
#             add_question_chars_idxs.append(add_new_question_char_idxs.tolist())
#             add_y1s.append(new_y1.tolist())
#             add_y2s.append(new_y2.tolist())
#             add_ids.append(ids[idx].item() + last_id)
#         print("finished batch")

        
#     print("finished creating all tensors")

#     context_idxs = torch.cat((context_idxs, torch.tensor(add_context_idxs)), 0)
#     context_char_idxs = torch.cat((context_char_idxs, torch.tensor(add_context_chars_idxs)), 0)

#     question_idxs = torch.cat((question_idxs, torch.tensor(add_question_idxs)), 0)
#     question_char_idxs = torch.cat((question_char_idxs, torch.tensor(add_question_chars_idxs)), 0)  

#     y1s = torch.cat((y1s, torch.tensor(add_y1s)), 0)
#     y2s = torch.cat((y2s, torch.tensor(add_y2s)), 0)

#     ids = torch.cat((ids, torch.tensor(add_ids)), 0)  
#     # save into a new file
#     print("Finished processing all training examples. Saving into file.")
#     np.savez_compressed(output_path, 
#                         context_idxs=context_idxs, 
#                         context_char_idxs=context_char_idxs, 
#                         ques_idxs=question_idxs, 
#                         ques_char_idxs=question_char_idxs, 
#                         y1s=y1s, 
#                         y2s=y2s, 
#                         ids=ids)

#     print("Finished saving data into file.")


def collate_fn(examples):
    """Create batch tensors from a list of individual examples returned
    by `SQuAD.__getitem__`. Merge examples of different length by padding
    all examples to the maximum length in the batch.

    Args:
        examples (list): List of tuples of the form (context_idxs, context_char_idxs,
        question_idxs, question_char_idxs, y1s, y2s, ids).

    Returns:
        examples (tuple): Tuple of tensors (context_idxs, context_char_idxs, question_idxs,
        question_char_idxs, y1s, y2s, ids). All of shape (batch_size, ...), where
        the remaining dimensions are the maximum length of examples in the input.

    Adapted from:
        https://github.com/yunjey/seq2seq-dataloader
    """
    def merge_0d(scalars, dtype=torch.int64):
        return torch.tensor(scalars, dtype=dtype)

    def merge_1d(arrays, dtype=torch.int64, pad_value=0):
        lengths = [(a != pad_value).sum() for a in arrays]
        padded = torch.zeros(len(arrays), max(lengths), dtype=dtype)
        for i, seq in enumerate(arrays):
            end = lengths[i]
            padded[i, :end] = seq[:end]
        return padded

    def merge_2d(matrices, dtype=torch.int64, pad_value=0):
        heights = [(m.sum(1) != pad_value).sum() for m in matrices]
        widths = [(m.sum(0) != pad_value).sum() for m in matrices]
        padded = torch.zeros(len(matrices), max(heights), max(widths), dtype=dtype)
        for i, seq in enumerate(matrices):
            height, width = heights[i], widths[i]
            padded[i, :height, :width] = seq[:height, :width]
        return padded

    # Group by tensor type
    context_idxs, context_char_idxs, \
        question_idxs, question_char_idxs, \
        y1s, y2s, ids = zip(*examples)

    # Merge into batch tensors
    context_idxs = merge_1d(context_idxs)
    context_char_idxs = merge_2d(context_char_idxs)
    question_idxs = merge_1d(question_idxs)
    question_char_idxs = merge_2d(question_char_idxs)
    y1s = merge_0d(y1s)
    y2s = merge_0d(y2s)
    ids = merge_0d(ids)

    return (context_idxs, context_char_idxs,
            question_idxs, question_char_idxs,
            y1s, y2s, ids)


class AverageMeter:
    """Keep track of average values over time.

    Adapted from:
        > https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        """Reset meter."""
        self.__init__()

    def update(self, val, num_samples=1):
        """Update meter with new value `val`, the average of `num` samples.

        Args:
            val (float): Average value to update the meter with.
            num_samples (int): Number of samples that were averaged to
                produce `val`.
        """
        self.count += num_samples
        self.sum += val * num_samples
        self.avg = self.sum / self.count


class EMA:
    """Exponential moving average of model parameters.
    Args:
        model (torch.nn.Module): Model with parameters whose EMA will be kept.
        decay (float): Decay rate for exponential moving average.
    """
    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = {}
        self.original = {}

        # Register model parameters
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def __call__(self, model, num_updates):
        decay = min(self.decay, (1.0 + num_updates) / (10.0 + num_updates))
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = \
                    (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        """Assign exponential moving average of parameter values to the
        respective parameters.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]

    def resume(self, model):
        """Restore original parameters to a model. That is, put back
        the values that were in each parameter at the last call to `assign`.
        Args:
            model (torch.nn.Module): Model to assign parameter values.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]


class CheckpointSaver:
    """Class to save and load model checkpoints.

    Save the best checkpoints as measured by a metric value passed into the
    `save` method. Overwrite checkpoints with better checkpoints once
    `max_checkpoints` have been saved.

    Args:
        save_dir (str): Directory to save checkpoints.
        max_checkpoints (int): Maximum number of checkpoints to keep before
            overwriting old ones.
        metric_name (str): Name of metric used to determine best model.
        maximize_metric (bool): If true, best checkpoint is that which maximizes
            the metric value passed in via `save`. Otherwise, best checkpoint
            minimizes the metric.
        log (logging.Logger): Optional logger for printing information.
    """
    def __init__(self, save_dir, max_checkpoints, metric_name,
                 maximize_metric=False, log=None):
        super(CheckpointSaver, self).__init__()

        self.save_dir = save_dir
        self.max_checkpoints = max_checkpoints
        self.metric_name = metric_name
        self.maximize_metric = maximize_metric
        self.best_val = None
        self.ckpt_paths = queue.PriorityQueue()
        self.log = log
        self._print(f"Saver will {'max' if maximize_metric else 'min'}imize {metric_name}...")

    def is_best(self, metric_val):
        """Check whether `metric_val` is the best seen so far.

        Args:
            metric_val (float): Metric value to compare to prior checkpoints.
        """
        if metric_val is None:
            # No metric reported
            return False

        if self.best_val is None:
            # No checkpoint saved yet
            return True

        return ((self.maximize_metric and self.best_val < metric_val)
                or (not self.maximize_metric and self.best_val > metric_val))

    def _print(self, message):
        """Print a message if logging is enabled."""
        if self.log is not None:
            self.log.info(message)

    def save(self, step, model, metric_val, device):
        """Save model parameters to disk.

        Args:
            step (int): Total number of examples seen during training so far.
            model (torch.nn.DataParallel): Model to save.
            metric_val (float): Determines whether checkpoint is best so far.
            device (torch.device): Device where model resides.
        """
        ckpt_dict = {
            'model_name': model.__class__.__name__,
            'model_state': model.cpu().state_dict(),
            'step': step
        }
        model.to(device)

        checkpoint_path = os.path.join(self.save_dir,
                                       f'step_{step}.pth.tar')
        torch.save(ckpt_dict, checkpoint_path)
        self._print(f'Saved checkpoint: {checkpoint_path}')

        if self.is_best(metric_val):
            # Save the best model
            self.best_val = metric_val
            best_path = os.path.join(self.save_dir, 'best.pth.tar')
            shutil.copy(checkpoint_path, best_path)
            self._print(f'New best checkpoint at step {step}...')

        # Add checkpoint path to priority queue (lowest priority removed first)
        if self.maximize_metric:
            priority_order = metric_val
        else:
            priority_order = -metric_val

        self.ckpt_paths.put((priority_order, checkpoint_path))

        # Remove a checkpoint if more than max_checkpoints have been saved
        if self.ckpt_paths.qsize() > self.max_checkpoints:
            _, worst_ckpt = self.ckpt_paths.get()
            try:
                os.remove(worst_ckpt)
                self._print(f'Removed checkpoint: {worst_ckpt}')
            except OSError:
                # Avoid crashing if checkpoint has been removed or protected
                pass


def load_model(model, checkpoint_path, gpu_ids, return_step=True):
    """Load model parameters from disk.

    Args:
        model (torch.nn.DataParallel): Load parameters into this model.
        checkpoint_path (str): Path to checkpoint to load.
        gpu_ids (list): GPU IDs for DataParallel.
        return_step (bool): Also return the step at which checkpoint was saved.

    Returns:
        model (torch.nn.DataParallel): Model loaded from checkpoint.
        step (int): Step at which checkpoint was saved. Only if `return_step`.
    """
    device = f"cuda:{gpu_ids[0]}" if gpu_ids else 'cpu'
    ckpt_dict = torch.load(checkpoint_path, map_location=device)

    # Build model, load parameters
    model.load_state_dict(ckpt_dict['model_state'])

    if return_step:
        step = ckpt_dict['step']
        return model, step

    return model


def get_available_devices():
    """Get IDs of all available GPUs.

    Returns:
        device (torch.device): Main device (GPU 0 or CPU).
        gpu_ids (list): List of IDs of all GPUs that are available.
    """
    gpu_ids = []
    if torch.cuda.is_available():
        gpu_ids += [gpu_id for gpu_id in range(torch.cuda.device_count())]
        device = torch.device(f'cuda:{gpu_ids[0]}')
        torch.cuda.set_device(device)
    else:
        device = torch.device('cpu')

    return device, gpu_ids


def masked_softmax(logits, mask, dim=-1, log_softmax=False):
    """Take the softmax of `logits` over given dimension, and set
    entries to 0 wherever `mask` is 0.

    Args:
        logits (torch.Tensor): Inputs to the softmax function.
        mask (torch.Tensor): Same shape as `logits`, with 0 indicating
            positions that should be assigned 0 probability in the output.
        dim (int): Dimension over which to take softmax.
        log_softmax (bool): Take log-softmax rather than regular softmax.
            E.g., some PyTorch functions such as `F.nll_loss` expect log-softmax.

    Returns:
        probs (torch.Tensor): Result of taking masked softmax over the logits.
    """
    mask = mask.type(torch.float32)
    masked_logits = mask * logits + (1 - mask) * -1e30
    softmax_fn = F.log_softmax if log_softmax else F.softmax
    probs = softmax_fn(masked_logits, dim)

    return probs


def visualize(tbx, pred_dict, eval_path, step, split, num_visuals):
    """Visualize text examples to TensorBoard.

    Args:
        tbx (tensorboardX.SummaryWriter): Summary writer.
        pred_dict (dict): dict of predictions of the form id -> pred.
        eval_path (str): Path to eval JSON file.
        step (int): Number of examples seen so far during training.
        split (str): Name of data split being visualized.
        num_visuals (int): Number of visuals to select at random from preds.
    """
    if num_visuals <= 0:
        return
    if num_visuals > len(pred_dict):
        num_visuals = len(pred_dict)

    visual_ids = np.random.choice(list(pred_dict), size=num_visuals, replace=False)

    with open(eval_path, 'r') as eval_file:
        eval_dict = json.load(eval_file)
    for i, id_ in enumerate(visual_ids):
        pred = pred_dict[id_] or 'N/A'
        example = eval_dict[str(id_)]
        question = example['question']
        context = example['context']
        answers = example['answers']

        gold = answers[0] if answers else 'N/A'
        tbl_fmt = (f'- **Question:** {question}\n'
                   + f'- **Context:** {context}\n'
                   + f'- **Answer:** {gold}\n'
                   + f'- **Prediction:** {pred}')
        tbx.add_text(tag=f'{split}/{i+1}_of_{num_visuals}',
                     text_string=tbl_fmt,
                     global_step=step)


def save_preds(preds, save_dir, file_name='predictions.csv'):
    """Save predictions `preds` to a CSV file named `file_name` in `save_dir`.

    Args:
        preds (list): List of predictions each of the form (id, start, end),
            where id is an example ID, and start/end are indices in the context.
        save_dir (str): Directory in which to save the predictions file.
        file_name (str): File name for the CSV file.

    Returns:
        save_path (str): Path where CSV file was saved.
    """
    # Validate format
    if (not isinstance(preds, list)
            or any(not isinstance(p, tuple) or len(p) != 3 for p in preds)):
        raise ValueError('preds must be a list of tuples (id, start, end)')

    # Make sure predictions are sorted by ID
    preds = sorted(preds, key=lambda p: p[0])

    # Save to a CSV file
    save_path = os.path.join(save_dir, file_name)
    np.savetxt(save_path, np.array(preds), delimiter=',', fmt='%d')

    return save_path


def get_save_dir(base_dir, name, training, id_max=100):
    """Get a unique save directory by appending the smallest positive integer
    `id < id_max` that is not already taken (i.e., no dir exists with that id).

    Args:
        base_dir (str): Base directory in which to make save directories.
        name (str): Name to identify this training run. Need not be unique.
        training (bool): Save dir. is for training (determines subdirectory).
        id_max (int): Maximum ID number before raising an exception.

    Returns:
        save_dir (str): Path to a new directory with a unique name.
    """
    for uid in range(1, id_max):
        subdir = 'train' if training else 'test'
        save_dir = os.path.join(base_dir, subdir, f'{name}-{uid:02d}')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            return save_dir

    raise RuntimeError('Too many save directories created with the same name. \
                       Delete old save directories or use another name.')


def get_logger(log_dir, name):
    """Get a `logging.Logger` instance that prints to the console
    and an auxiliary file.

    Args:
        log_dir (str): Directory in which to create the log file.
        name (str): Name to identify the logs.

    Returns:
        logger (logging.Logger): Logger instance for logging events.
    """
    class StreamHandlerWithTQDM(logging.Handler):
        """Let `logging` print without breaking `tqdm` progress bars.

        See Also:
            > https://stackoverflow.com/questions/38543506
        """
        def emit(self, record):
            try:
                msg = self.format(record)
                tqdm.tqdm.write(msg)
                self.flush()
            except (KeyboardInterrupt, SystemExit):
                raise
            except:
                self.handleError(record)

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Log everything (i.e., DEBUG level and above) to a file
    log_path = os.path.join(log_dir, 'log.txt')
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)

    # Log everything except DEBUG level (i.e., INFO level and above) to console
    console_handler = StreamHandlerWithTQDM()
    console_handler.setLevel(logging.INFO)

    # Create format for the logs
    file_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                       datefmt='%m.%d.%y %H:%M:%S')
    file_handler.setFormatter(file_formatter)
    console_formatter = logging.Formatter('[%(asctime)s] %(message)s',
                                          datefmt='%m.%d.%y %H:%M:%S')
    console_handler.setFormatter(console_formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


def torch_from_json(path, dtype=torch.float32):
    """Load a PyTorch Tensor from a JSON file.

    Args:
        path (str): Path to the JSON file to load.
        dtype (torch.dtype): Data type of loaded array.

    Returns:
        tensor (torch.Tensor): Tensor loaded from JSON file.
    """
    with open(path, 'r') as fh:
        array = np.array(json.load(fh))

    tensor = torch.from_numpy(array).type(dtype)

    return tensor


def discretize(p_start, p_end, max_len=15, no_answer=False):
    """Discretize soft predictions to get start and end indices.

    Choose the pair `(i, j)` of indices that maximizes `p1[i] * p2[j]`
    subject to `i <= j` and `j - i + 1 <= max_len`.

    Args:
        p_start (torch.Tensor): Soft predictions for start index.
            Shape (batch_size, context_len).
        p_end (torch.Tensor): Soft predictions for end index.
            Shape (batch_size, context_len).
        max_len (int): Maximum length of the discretized prediction.
            I.e., enforce that `preds[i, 1] - preds[i, 0] + 1 <= max_len`.
        no_answer (bool): Treat 0-index as the no-answer prediction. Consider
            a prediction no-answer if `preds[0, 0] * preds[0, 1]` is greater
            than the probability assigned to the max-probability span.

    Returns:
        start_idxs (torch.Tensor): Hard predictions for start index.
            Shape (batch_size,)
        end_idxs (torch.Tensor): Hard predictions for end index.
            Shape (batch_size,)
    """
    if p_start.min() < 0 or p_start.max() > 1 \
            or p_end.min() < 0 or p_end.max() > 1:
        raise ValueError('Expected p_start and p_end to have values in [0, 1]')

    # Compute pairwise probabilities
    p_start = p_start.unsqueeze(dim=2)
    p_end = p_end.unsqueeze(dim=1)
    p_joint = torch.matmul(p_start, p_end)  # (batch_size, c_len, c_len)

    # Restrict to pairs (i, j) such that i <= j <= i + max_len - 1
    c_len, device = p_start.size(1), p_start.device
    is_legal_pair = torch.triu(torch.ones((c_len, c_len), device=device))
    is_legal_pair -= torch.triu(torch.ones((c_len, c_len), device=device),
                                diagonal=max_len)
    if no_answer:
        # Index 0 is no-answer
        p_no_answer = p_joint[:, 0, 0].clone()
        is_legal_pair[0, :] = 0
        is_legal_pair[:, 0] = 0
    else:
        p_no_answer = None
    p_joint *= is_legal_pair

    # Take pair (i, j) that maximizes p_joint
    max_in_row, _ = torch.max(p_joint, dim=2)
    max_in_col, _ = torch.max(p_joint, dim=1)
    start_idxs = torch.argmax(max_in_row, dim=-1)
    end_idxs = torch.argmax(max_in_col, dim=-1)

    if no_answer:
        # Predict no-answer whenever p_no_answer > max_prob
        max_prob, _ = torch.max(max_in_col, dim=-1)
        start_idxs[p_no_answer > max_prob] = 0
        end_idxs[p_no_answer > max_prob] = 0

    return start_idxs, end_idxs


def convert_tokens(eval_dict, qa_id, y_start_list, y_end_list, no_answer):
    """Convert predictions to tokens from the context.

    Args:
        eval_dict (dict): Dictionary with eval info for the dataset. This is
            used to perform the mapping from IDs and indices to actual text.
        qa_id (int): List of QA example IDs.
        y_start_list (list): List of start predictions.
        y_end_list (list): List of end predictions.
        no_answer (bool): Questions can have no answer. E.g., SQuAD 2.0.

    Returns:
        pred_dict (dict): Dictionary index IDs -> predicted answer text.
        sub_dict (dict): Dictionary UUIDs -> predicted answer text (submission).
    """
    pred_dict = {}
    sub_dict = {}
    for qid, y_start, y_end in zip(qa_id, y_start_list, y_end_list):
        context = eval_dict[str(qid)]["context"]
        spans = eval_dict[str(qid)]["spans"]
        uuid = eval_dict[str(qid)]["uuid"]
        if no_answer and (y_start == 0 or y_end == 0):
            pred_dict[str(qid)] = ''
            sub_dict[uuid] = ''
        else:
            if no_answer:
                y_start, y_end = y_start - 1, y_end - 1
            start_idx = spans[y_start][0]
            end_idx = spans[y_end][1]
            pred_dict[str(qid)] = context[start_idx: end_idx]
            sub_dict[uuid] = context[start_idx: end_idx]
    return pred_dict, sub_dict


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    if not ground_truths:
        return metric_fn(prediction, '')
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def eval_dicts(gold_dict, pred_dict, no_answer):
    avna = f1 = em = total = 0
    for key, value in pred_dict.items():
        total += 1
        ground_truths = gold_dict[key]['answers']
        prediction = value
        em += metric_max_over_ground_truths(compute_em, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(compute_f1, prediction, ground_truths)
        if no_answer:
            avna += compute_avna(prediction, ground_truths)

    eval_dict = {'EM': 100. * em / total,
                 'F1': 100. * f1 / total}

    if no_answer:
        eval_dict['AvNA'] = 100. * avna / total

    return eval_dict


def compute_avna(prediction, ground_truths):
    """Compute answer vs. no-answer accuracy."""
    return float(bool(prediction) == bool(ground_truths))


# All methods below this line are from the official SQuAD 2.0 eval script
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
def normalize_answer(s):
    """Convert to lowercase and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        regex = re.compile(r'\b(a|an|the)\b', re.UNICODE)
        return re.sub(regex, ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    if not s:
        return []
    return normalize_answer(s).split()


def compute_em(a_gold, a_pred):
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    common = Counter(gold_toks) & Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1
