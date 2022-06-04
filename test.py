import os, csv
import pandas as pd
from nltk import word_tokenize


def compute_accuracy(result_df, relation):
    total = result_df.shape[0]
    true = 0
    for _, row in result_df.iterrows():
        if (relation):
            if (str(row['Series']) in row['prediction']):
                true += 1
            if (row['prediction'] == '8'): 
                total -=1
        else: 
            if (row['Series'] == row['prediction']):
                true += 1
            if (row['prediction'] == 8): 
                total -=1
            
    accuracy = true/total
    return accuracy


input_file = pd.read_csv('./Data/test_case.csv')
text_list = input_file['Text']
text_type_list = input_file['sentence']
ambiguity_list = input_file['ambiguity']
# Sentence
# 0 : phrase
# 1 : clause
# 2 : sentence
# tuple : (average word number per each text, input number)

# Ambiguity
# 0 : unambiguous
# 1 : ambiguous to distinguish series
# 2 : ambiguous to distinguish novel
text_type = {'phrase': {'avg_word_num': 0, 'input_list': []}, 
            'clause': {'avg_word_num': 0, 'input_list': []}, 
            'sentence': {'avg_word_num': 0, 'input_list': []}}
ambiguity = [[], [], []]
for i, text in enumerate(text_list):
    if text_type_list[i] == 0:
        text_type['phrase']['avg_word_num'] += len(word_tokenize(text))
        text_type['phrase']['input_list'].append(i)
    elif text_type_list[i] == 1:
        text_type['clause']['avg_word_num'] += len(word_tokenize(text))
        text_type['clause']['input_list'].append(i)
    elif text_type_list[i] == 2:
        text_type['sentence']['avg_word_num'] += len(word_tokenize(text))
        text_type['sentence']['input_list'].append(i)

    if ambiguity_list[i] == 0:
        ambiguity[0].append(i)
    elif ambiguity_list[i] == 1:
        ambiguity[1].append(i)
    elif ambiguity_list[i] == 2:
        ambiguity[2].append(i)
text_type['phrase']['avg_word_num'] = text_type['phrase']['avg_word_num'] / len(text_type['phrase']['input_list'])
text_type['clause']['avg_word_num'] = text_type['clause']['avg_word_num'] / len(text_type['clause']['input_list'])
text_type['sentence']['avg_word_num'] = text_type['sentence']['avg_word_num'] / len(text_type['sentence']['input_list'])

# print(text_type)
# print()
# print(ambiguity)

print("number info for each input type: {} / {} / {}".format(len(text_type['phrase']['input_list']), len(text_type['clause']['input_list']), len(text_type['sentence']['input_list'])))
print("number info for each ambiguity type: {} / {} / {}".format(len(ambiguity[0]), len(ambiguity[1]), len(ambiguity[2])))





relation_result_df = pd.read_csv('./Data/results/test_result_relation.csv')
no_relation_result_df = pd.read_csv('./Data/results/test_result_no_relation.csv')


relation_accuracy = compute_accuracy(relation_result_df, relation=True)
no_relation_accuracy = compute_accuracy(no_relation_result_df, relation=False)
print("using relation method accuracy: ", relation_accuracy*100)
print("not using relation method accuracy: ", no_relation_accuracy*100)
# total = len(text_type['phrase']['input_list'])
# total = len(ambiguity[1])


# TF_IDF
# TOTAL: 46.07843137254902 (47/102)
# According to Type
# Phrase: 26.31578947368421% (5/19) / Clause: 27.27272727272727% (6/22) / Sentence: 45.56962025316456 (36/79)
# According to ambiguity
# 0 : 42.2680412371134(41/97) / 1: 27.77777777777778(5/18) / 2: 20.0(1/5)


#Relation
#TOTAL: 47.05882352941176 % (48/102)
#Phrase: 26.31578947368421(5/19)/ Clause: 31.818181818181817% (7/22)/ Sentence: 45.56962025316456(36/79)
#According to ambiguity
# 0: 41.23711340206185(40/97) / 1: 38.88888888888889(7/18) / 2: 20.0 (1/5)
