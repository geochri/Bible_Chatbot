# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 17:11:41 2019

@author: WT
"""

import pandas as pd
import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.corpus import conll2000
from gensim.models import Word2Vec

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
    documents = []
    for document in df["t"]:
        sentences = nltk.sent_tokenize(document)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [[word.lower() for word in sent if word.isalnum()] for sent in sentences]
        documents.extend(sentences)
    return documents

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

datafolder = "./data/"
df = pd.read_csv(os.path.join(datafolder,"t_bbe.csv"))

ecc = df[df["b"]==21]
documents = ie_preprocess(ecc)
train_sents = conll2000.chunked_sents("train.txt", chunk_types=["NP"])
bigram_chunker = BigramChunker(train_sents)
sent_processed = tag_chunk_documents(documents, bigram_chunker)
sent_tree = convert_sentprocessed_to_tree(sent_processed)
noun_phrases = get_noun_phrases(sent_tree)

documents_token = preprocessW2V(ecc)
model = Word2Vec()

'''
vectorizer = CountVectorizer(input="content", max_features=1000000)
dtm = vectorizer.fit_transform(df["t"])
vocab = vectorizer.get_feature_names()
dtm = dtm.toarray()
vocab = np.array(vocab)

counts = np.sum(dtm, axis=0)
counts_idx = counts.argsort()[-1000:][::-1]
stop_words = list(vocab[counts_idx])

##### REMOVE STOPWORDS top 200
vectorizer = CountVectorizer(input="content", stop_words=stop_words, max_features=1000000)
dtm = vectorizer.fit_transform(df["t"])
vocab = vectorizer.get_feature_names()
dtm = dtm.toarray()
vocab = np.array(vocab)

df_bow = pd.DataFrame(data=dtm, index=["Doc_"+str(i) for i in range(len(dtm))],\
                        columns=list(vocab))
col_list = list(df_bow.columns)
# DO LDA
lda = LatentDirichletAllocation(n_components=10, random_state=0, n_jobs=-1)
df_bow = lda.fit_transform(df_bow)
words_per_topic = lda.components_
top5words_pertopic = []
print("Top 10 words per topic:")
for idx,topic in enumerate(words_per_topic):
    #top5values = topic.argsort()[-5:][::-1]
    top5idx = (-topic).argsort()[:10]
    top5words = [col_list[widx] for widx in top5idx]
    top5words_pertopic.append(top5words)
    print(f"Topic {idx}: ", top5words)

print("Topic distribution per document")
for didx,doc in enumerate(df_bow):
    print(f"Doc {didx}:")
    for tidx,topic in enumerate(doc):
        print(f"Topic {tidx} : {topic*100} %")
    if didx == 10:
        break
'''
