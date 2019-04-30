# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:11:41 2019

@author: WT
"""
############### Script that contains modules for chatbot ##############

import numpy as np
import nltk
import os
import pickle
import re
from thesaurus import Word
import requests
from bs4 import BeautifulSoup

def save_as_pickle(filename, data):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'wb') as output:
        pickle.dump(data, output)
        
def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data
        
## tokenizes and attaches pos tags to sentences in df
def ie_preprocess(df):
    documents = []
    for document in df["t"]:
        sentences = nltk.sent_tokenize(document)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [nltk.pos_tag(sent) for sent in sentences]
        documents.extend(sentences)
    return documents

def preprocessW2V(df):
    documents = []; documents_raw = []
    for document in df["t"]:
        sentences = nltk.sent_tokenize(document); documents_raw.extend(sentences)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [[word.lower() for word in sent if word.isalnum()] for sent in sentences]
        documents.extend(sentences)
    return documents_raw, documents

## chunks documents based on grammar1 exp
def chunker(documents):
    grammar1 = r"""NP: {<DT|PRP.?>?<JJ.?>*<NN.*>}
                {<NNP>+}
                PP: {<IN><NP>}
                VP: {<VB.*><NP|PP|CLAUSE>+$}
                CLAUSE: {<NP><VP>}"""
    cp = nltk.RegexpParser(grammar1, loop=2)
    chunked_trees = []
    for sent in documents:
        chunked_trees.append(cp.parse(sent))
    return chunked_trees

## chunker based on bigram trained on conll2000 corpus
class BigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): #list of chunk sent tree
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)] for \
                       sent in train_sents]
        self.tagger = nltk.BigramTagger(train_data)
    
    def parse(self, sentence):
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        colltags = [(word, pos, chunktag) for ((word,pos),chunktag) in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(colltags)

def npchunk_features(untagged_sent, i, history):
    word, pos = untagged_sent[i]
    if i == 0:
        prevword, prevpos = "<START>", "<START>"
    else:
        prevword, prevpos = untagged_sent[i-1]
    return {"pos": pos, "word": word, "prevpos": prevpos}

class ConsecutiveNPChunkTagger(nltk.TaggerI):
    def __init__(self, train_sents): # 
        train_set = []
        for tagged_sent in train_sents:
            untagged_sent = nltk.tag.untag(tagged_sent)
            history = []
            for i, (word, tag) in enumerate(tagged_sent):
                featureset = npchunk_features(untagged_sent, i, history)
                train_set.append((featureset, tag))
                history.append(tag)
        self.classifier = nltk.MaxentClassifier.train(train_set, trace=0)

    def tag(self, sentence):
        history = []
        for i, word in enumerate(sentence):
            featureset = npchunk_features(sentence, i, history)
            tag = self.classifier.classify(featureset)
            history.append(tag)
        return zip(sentence, history)

class ConsecutiveNPChunker(nltk.ChunkParserI):
    def __init__(self, train_sents):
        tagged_sents = [[((w,t),c) for (w,t,c) in
                         nltk.chunk.tree2conlltags(sent)]
                        for sent in train_sents]
        self.tagger = ConsecutiveNPChunkTagger(tagged_sents)

    def parse(self, sentence):
        tagged_sents = self.tagger.tag(sentence)
        conlltags = [(w,t,c) for ((w,t),c) in tagged_sents]
        return nltk.chunk.conlltags2tree(conlltags)

def tag_chunk_documents(documents, bigram_chunker):
    sent_processed = []
    for sent in documents:
        posses = [pos for (w, pos) in sent]
        posses = bigram_chunker.tagger.tag(posses)
        posses = [c for (pos, c) in posses]
        sent_processed.append([(w, pos, c) for (w,pos), c in zip(sent, posses)])
    return sent_processed

def convert_sentprocessed_to_tree(sent_processed):
    sent_tree = []
    for sent in sent_processed:
        sent_tree.append(nltk.conlltags2tree(sent))
    return sent_tree

# get all noun phrases
def get_noun_phrases(sent_tree):
    noun_phrases = []
    for tree in sent_tree:
        for subtree in tree.subtrees(filter=lambda x:x.label() == "NP"):
            noun_phrases.append(" ".join(word for word, tag in subtree.leaves()))
    noun_phrases = list(set(noun_phrases))
    return noun_phrases

# vectorize query sentences
def vectorize_sent(documents_token, model):
    sents_vec = []
    vocab = model.wv.vocab
    for sent in documents_token:
        l = 0; wv = 0
        for token in sent:
            if token in vocab:
                wv += model.wv[token]
                l += 1
            else: # check for synonyms if word not in vocab
                try:
                    syns = Word(token).synonyms()
                    for syn in syns:
                        if syn in vocab:
                            wv += model.wv[syn]
                            l += 1; break
                except:
                    continue
        if l != 0:
            sents_vec.append(wv/l)
        else:
            sents_vec.append(None)
    return sents_vec

# find most similar sentence
def most_sim_sent(query, sents_vec):
    query = np.array(query)
    score = []
    for s in sents_vec:
        s = np.array(s)
        try:
            score.append(np.dot(s,query[0])/(np.dot(s,s)*np.dot(query[0],query[0])))
        except:
            score.append(0)
    score = np.array(score)
    if all(score == 0):
        return score, score.argsort()[-np.random.randint(2,5):][::-1], False
    else:
        return score, score.argsort()[-np.random.randint(2,5):][::-1], True

def user_query(query, model, sents_vec, documents_raw, stopwords):
    query = nltk.word_tokenize(query.lower())
    #query = [word.lower() for word in query if word.isalnum()]
    query = [word for word in query if word not in stopwords]
    query = vectorize_sent([query], model)
    sim_sent_score, sim_sent_idx, got_match = most_sim_sent(query, sents_vec)
    if got_match == True:
        ans = "\n".join([documents_raw[idx] for idx in sim_sent_idx])
        return ans
    else:
        return np.random.choice(["Have peace. God loves you.", "Don't fret, we are all God's children, He will protect us.",\
                                 "Don't worry, nothing is impossible with God.", "Be at peace with God."])

class user_profile():
    def __init__(self):
        super(user_profile,self).__init__()
        self.name = ""
        self.recipient_id = None
        self.gender = ""
        self.age = None
        self.interests = []
        self.annoyance = 0
        self.sad = 0
        
    def save(self):
        save_as_pickle("profiles/profile_%s.profile" % str(self.recipient_id), self)

def get_name(text, bigram_chunker):
    stop_nouns = ["i", "he", "she", "they", "them", "it", "my", "me","we","you","a","the","an",\
                  "who","what","why","when","how"]
    #bigram_chunker = load_pickle("bigram_chunker.pth.tar")
    sent = nltk.pos_tag(nltk.word_tokenize(text))
    sent_processed = tag_chunk_documents([sent], bigram_chunker)
    names = []
    for i, w in enumerate(sent_processed[0]):
        dummy = []
        if w[2] == "B-NP":
            dummy = w[0]
            for next_w in sent_processed[0][(i+1):]:
                if next_w[2] == "I-NP":
                    dummy = dummy + " " + next_w[0]
                else:
                    break
        if dummy != []:
            flag = 0; dummies = dummy.lower().split()
            for d in dummies:
                if d in stop_nouns:
                    flag = 1
            if flag == 0:
                names.append(dummy)
    if names != []:
        return names[0]
    else:
        return None
    
def get_gender(text):
    m = ["male", "boy", "m", "guy", "dude", "man", "gentleman", "bloke", "hunk", "mr", "uncle"]
    f = ["female", "girl", "woman", "f", "lady", "madam", "mrs", "miss", "auntie", "auntie"] 
    words = nltk.word_tokenize(text); words = [w.lower() for w in words]
    for w in words:
        if w in m:
            return "male"
        if w in f:
            return "female"
    return

def get_age(text):
    ages = {'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 'seven': 7, \
            'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13, \
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17, 'eighteen': 18, \
            'nineteen': 19, 'twenty': 20, 'thirty': 30, 'forty': 40, 'fifty': 50, 'sixty': 60, \
            'seventy': 70, 'eighty': 80, 'ninety': 90, 'one hundred': 100}
    age = re.findall("\d+", text)
    if age != []:
        age = age[0]
    ### not a numberic integer? Try find numerical words
    if age == []:
        words = nltk.word_tokenize(text.replace("-", " "))
        age_w = []
        for w in words:
            if w in ages:
                age_w.append(ages[w])
        age = sum(age_w)
        if age == 0:
            return
    return age

#bigram_chunker = load_pickle("bigram_chunker.pth.tar")
def get_interests(text, bigram_chunker):
    stop_interests = ["i", "he", "she", "they", "them", "it", "my", "me","we","you","a","the","an",\
                      "who","what","why","when","how"]
    sent = nltk.pos_tag(nltk.word_tokenize(text))
    sent_processed = tag_chunk_documents([sent], bigram_chunker)
    interests = []; real_interests = []
    for i, w in enumerate(sent_processed[0]):
        dummy = []
        # look for noun phrases
        if w[2] == "B-NP":
            dummy = w[0]
            for next_w in sent_processed[0][(i+1):]:
                if next_w[2] == "I-NP":
                    dummy = dummy + " " + next_w[0]
                else:
                    break
            interests.append(dummy)
        # look for action verbs
        if w[1] == "VBG":
            interests.append(w[0])
        #### filter out those in stop_interests
    if interests != []:
        for i in interests:
            flag = 0; i_s = i.lower().split()
            for w in i_s:
                if w in stop_interests:
                    flag = 1
            if flag == 0:
                real_interests.append(i)
    if real_interests != []:
        return real_interests
    else:
        return None

def get_verse(usertext, book_dict):
    chap = ["chronicles", "corinthians", "john", "kings", "peter", "samuel", "thessalonians", "timothy",]
    book, chapter, verse, verse_end = None, None, None, None
    text = nltk.word_tokenize(usertext.lower())
    for c in chap:
        s = re.search("(\d+ +%s)" % c, usertext)
        if s != None:
            book = s.group(0); book = book_dict[book]
            break
    if book == None:
        for w in text:
            if w in book_dict.keys():
                book = book_dict[w]
                break
    verse = re.search("verse +(\d+)", usertext)
    chapter = re.search("chapter +(\d+)", usertext)
    if verse != None:
        verse = verse.group(1)
    if chapter != None:
        chapter = chapter.group(1)
    # \d:\d-\d format
    if (verse == None) and (chapter == None):
        s = re.search("(\d+):(\d+) *-+ *(\d+)", usertext)
        if s != None:
            chapter = s.group(1); verse = s.group(2); verse_end = s.group(3)
    # \d:\d format
    if (verse == None) and (chapter == None) and (verse_end == None):
        s = re.search("(\d+):(\d+)", usertext)
        if s != None:
            chapter = s.group(1); verse = s.group(2)
    return book, chapter, verse, verse_end

def get_content(url):
    user_agent = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Safari/537.36"}
    resp = requests.get(url, headers=user_agent)
    if resp.ok:
        return resp.content

# searches gotquestions.org for query and returns most relevant answer
def get_answers(usertext):
    text = nltk.word_tokenize(usertext.lower())
    text = "+".join(w for w in text); # print(text)
    url = "https://www.gotquestions.org/search.php?zoom_query=%s" % text
    soup = BeautifulSoup(get_content(url), "html.parser")
    top_url = soup.find_all("div", class_= "infoline")[0].get_text()[5:]
    soup = BeautifulSoup(get_content(top_url), "html.parser")
    ans = soup.find_all("span", itemprop = "articleBody")[0].get_text()
    l_ans = len(ans); answer = []; i = 0
    while (i < l_ans):
        answer.append(ans[i:(i+3000)])
        i += 3000
    return answer