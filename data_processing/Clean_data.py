# -*- coding: utf-8 -*-
import re
from string import punctuation
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
import unicodedata

repeat_regexp = re.compile(r'(\w*)(\w)\2(\w*)')
repl = r'\1\2\3'

stemmer = SnowballStemmer('spanish')
# Sign punctuation
non_words = list(punctuation)
# Sign spanish
non_words.extend(['¿', '¡', '“', '¡'])
# Sign number and other sign
non_words.extend(map(str, range(10)))


def elimina_tildes(s=''):
    # if isinstance(s, str):
    #     s = s.decode('utf-8')
    s = ''.join((c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn'))
    return s


spanish_stopwords = stopwords.words('spanish')
spanish_stopwords.extend(['htp', 'que', 'ahi', 'como', 'cual', 'donde', 'porque'])
spanish_stopwords = list(map(lambda x: stemmer.stem(elimina_tildes(x)), spanish_stopwords))
spanish_stopwords.extend(['estuvi', 'hubi', 'tuvi'])

x = 1


def clean(text):
    text = elimina_tildes(text)
    # Replace retweet mentioned in the text with the string literal RT
    text = re.sub('(RT|via)((?:\\b\\W*@\\w+)+)', ' ', text)
    # Replace usernames mentioned in the text with the string literal USER
    # text = re.sub('@\\w+', text[1:], text)
    # Replace URLs with the string URL
    text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    # remove htt...
    text = re.sub("ht[t]?…", "", text)
    # Delete numbers
    text = re.sub(r'[0-9]+', '', text)
    # Replace newline
    text = text.replace('\n', ' ')
    # Replace tab
    text = text.replace('\t', ' ')
    # Replace r
    text = text.replace('\r', ' ')
    # Removal of Punctuations
    text = re.sub('[%s]' % re.escape(punctuation), ' ', text)
    text = text.replace('¿', ' ')
    text = text.replace('–', ' ')
    text = text.replace('—', ' ')
    text = text.replace('“', ' ')
    text = text.replace('”', ' ')
    text = text.replace('…', ' ')
    text = text.replace('¡', ' ')

    # Convierte a minuscula
    text = text.lower()
    text = text.strip()
    return text


def replace(word):
    if word.find('rr') >= 0 or word.find('ll') >= 0:
        return word
    repl_word = repeat_regexp.sub(repl, word)
    if repl_word != word:
        return replace(repl_word)
    else:
        return repl_word


def stem_tokens(tokens):
    stemmed = []
    for item in tokens:
        item = replace(item)
        if 2 < len(item) < 24:
            stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    # print text
    try:
        text = text
        text = ''.join([c for c in text if c not in non_words])
        tokens = word_tokenize(text)
        # stem
        stems = stem_tokens(tokens)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems


def split_into_tokens(text):
    text = ''.join([c for c in text if c not in non_words])
    tokens = word_tokenize(text)
    return tokens


def split_into_lemmas(text):
    text = ''.join([c for c in text if c not in non_words])
    tokens = word_tokenize(text)
    # stem
    try:
        stems = stem_tokens(tokens)
    except Exception as e:
        print(e)
        print(text)
        stems = ['']
    return stems


def update_text(text=""):
    return text.replace('"', '').replace('Ã¡', 'á').replace('Ã©', 'é').replace('Ã­', 'í').replace('Ã³', 'ó') \
        .replace('Ãº', 'ú').replace('Ã±', 'ñ').replace('Ã¹', 'ù').replace('Â¡', '¡') \
        .replace('Â¬', '¬').replace('Â¿', '¿').replace('âˆ¼', '∼').replace('^M', ' ').replace('\n', ' ') \
        .replace('Ãš', 'ú').replace('Ã‘', 'ñ').replace('\r', ' ').strip()
