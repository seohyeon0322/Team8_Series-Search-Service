import os
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from text_processing import compute_tfidf, get_corpus
from nltk.parse import stanford

path_jar = './jars/stanford-parser.jar'
path_models_jar = './jars/stanford-corenlp-4.2.0-models-english.jar'
dep_parser = stanford.StanfordDependencyParser(path_to_jar = path_jar, path_to_models_jar = path_models_jar)


def dependency_parsing(sentence):
    wnl = WordNetLemmatizer()
    try:
        result = dep_parser.raw_parse(sentence)
    except:
        return (None, None ,None)
    else:
        dependency = result.__next__()
        nodes = dependency.nodes
        relation_triples = []
        for i in range (1, len(nodes)):
            subj = None
            obj = None
            triple = tuple()
            if (nodes[i]['rel'] == 'root' and nodes[i]['tag'].startswith('VB')):
                for dep in list(nodes[i]['deps'].keys()):
                    if (dep == 'nsubj'):
                        subj_index = nodes[i]['deps']['nsubj'][0]
                        subj = wnl.lemmatize(nodes[subj_index]['word'],  get_wordnet_pos(nodes[i]['tag']))
                        subj = subj.lower()
                    elif (dep == 'obj'):
                        obj_index = nodes[i]['deps']['obj'][0]
                        obj = wnl.lemmatize(nodes[obj_index]['word'],  get_wordnet_pos(nodes[i]['tag']))
                        obj = obj.lower()
                root = wnl.lemmatize(nodes[i]['word'], get_wordnet_pos(nodes[i]['tag']))
                root = root.lower()
                triple = (subj, root, obj)
                relation_triples.append(triple)
    return relation_triples


def get_wordnet_pos(treebank_tag):

    if treebank_tag.startswith('J'):
        return 'a'
    elif treebank_tag.startswith('V'):
        return 'v'
    elif treebank_tag.startswith('N'):
        return 'n'
    elif treebank_tag.startswith('R'):
        return 'r'
    else:
        return ''


def series_voca_set(sent):
    
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(sent)
    word_set = [word.lower() for word in words if word.lower() not in stop_words]
    wnl = WordNetLemmatizer()

    lemma_set = []
    for word, tag in pos_tag(word_set):
        wn_tag = get_wordnet_pos(tag)
        if wn_tag == "":
            lemma_set.append(wnl.lemmatize(word))
        else:
            lemma_set.append(wnl.lemmatize(word, wn_tag))

    return lemma_set


def get_tfidf(novel_data, input):
    result = dict()
    for word in input:
        if (word in novel_data.columns):
            result[word]=novel_data[word].values
    return result


def scaling(score_list):
    total = sum(j for (i, j) in score_list)
    result_list = []
    for (series, score) in score_list:
        score = score/total
        result_list.append((series, score))
    return result_list


def extract_relation(input_text):
    path = './Data/relations/'

    result = dependency_parsing(input_text)
    answer_list = [0,0,0,0,0,0,0]
    answer = 0

    series_list = sorted(os.listdir(path))
    if (series_list[0] == '.DS_Store'):
        del series_list[0]
    series_num = 0
    if (len(result) !=0):
        (subj, verb, obj) = result[0]
        for series in series_list:
            relations = pd.read_csv(path+series)
            for (index, row) in relations.iterrows():
                if ((row['0'] == subj and row['1'] == verb) 
                or (row['1'] == verb and row['2'] == obj)
                or (row['1'] == subj and row['2'] == obj)):
                        answer_list[series_num] += 1
            series_num +=1

    answer = [i+1 for i, j in enumerate(answer_list) if (j == max(answer_list) and j!=0)]
    if (len(answer) == 0):
        answer = 0
    else:
        answer = " ".join (str(i) for i in answer)
    return answer

def get_scores(corpus, input): 
    text = corpus.copy()
    lemmas = series_voca_set(input)
    input_lemma = " ".join(lemmas)
    text.insert(0, input_lemma)
    tfidf, tfidv = compute_tfidf(text)
    cosine_matrix = cosine_similarity(tfidf, tfidf)
    sim_scores = [(i, c) for i, c in enumerate(cosine_matrix[0]) if i != 0] 
    sim_scores = sorted(sim_scores, key = lambda x: x[1], reverse=True) 
    return sim_scores

def processing_score(input, scores, relation):
    if (scores[0][1]<0.02): #not related 
        return 8
    else: 
        scaled_scores = scaling(scores)
        if (scaled_scores[0][1] < 0.2): 
            if (relation):
                answer = extract_relation(input)
                if (answer == 0):
                    answer = scores[0][0]
            else:
                answer = scores[0][0]
            return answer
        else: 
            return scaled_scores[0][0]


"""
Below codes: to save csv file for compute accuracy
"""

# test_file = pd.read_csv('./Data/test_case.csv')

# relation_answers = []
# no_relation_answers = []

# book = 'Harry Potter'
# corpus = get_corpus(book)

# for index, row in test_file.iterrows():
#     text = corpus.copy()
#     input = row['Text']

#     sim_scores = get_scores(corpus, input)

#     relation_answer = processing_score(input, sim_scores, relation=True)
#     relation_answers.append(relation_answer)

#     no_relation_answer = processing_score(input, sim_scores, relation=False)
#     no_relation_answers.append(no_relation_answer)

# relation_answer_df = pd.DataFrame(relation_answers, columns=['prediction'])
# relation_test_result = pd.concat([test_file, relation_answer_df], axis = 1) 
# relation_test_result.to_csv("./Data/results/test_result_relation.csv", mode='w')


# no_relation_answer_df = pd.DataFrame(no_relation_answers, columns=['prediction'])
# no_relation_test_result = pd.concat([test_file, no_relation_answer_df], axis = 1) 
# no_relation_test_result.to_csv("./Data/results/test_result_no_relation.csv", mode='w')

