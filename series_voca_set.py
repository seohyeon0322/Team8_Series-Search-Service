import os
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

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

def series_voca_set(file):
    novel = file.read()
    words = word_tokenize(novel)
    
    stop_words = set(stopwords.words("english"))

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


novels = os.listdir("./Data/Harry Potter/")
for idx, series in enumerate(novels):
    f = open("./Data/Harry Potter/" + series, 'r', encoding='latin-1')
    word_set = ' '.join(sorted(series_voca_set(f)))
    f.close()

    file = open('./Data/novel_wordset/{}/series {}.txt'.format('Harry Potter', idx + 1), 'w', encoding='latin-1')
    file.write(word_set)
    file.close()