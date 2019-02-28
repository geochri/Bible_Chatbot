# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:13:09 2018

@author: elelswt
"""
import urllib.request as req
import random
from flask import Flask, request
from pymessenger.bot import Bot
from gensim.models import Word2Vec
import nltk
import pandas as pd
import os
from main_bible_chat import user_query
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms

app = Flask(__name__)
ACCESS_TOKEN = 'EAACfCUImm24BANefNtnE5hQgS4VEcu29c229gpXozaClVcq1JHKzezcZBnsnDADKf3yulAdYfo3ZB5MWZBJ5hFBAdXfx87hzFrcmFG0VUtOZCfRYKKzwlkEnpje8XsTIHjbVtyf2zYdyc4Ra0IFAYbs1MBuJH4ASJ2ZCHimngvEDIt1VbcZAmX'
VERIFY_TOKEN = 'tweetytweet!'
bot = Bot(ACCESS_TOKEN)

def preprocessW2V(df):
    documents = []; documents_raw = []
    for document in df["t"]:
        sentences = nltk.sent_tokenize(document); documents_raw.extend(sentences)
        sentences = [nltk.word_tokenize(sent) for sent in sentences]
        sentences = [[word.lower() for word in sent if word.isalnum()] for sent in sentences]
        documents.extend(sentences)
    return documents_raw, documents

# vectorize sentences
def vectorize_sent(documents_token, model):
    stopwords = list(set(nltk.corpus.stopwords.words("english")))
    sents_vec = []
    vocab = model.wv.vocab
    for sent in documents_token:
        l = 0; wv = 0
        for token in sent:
            if token in vocab:
                if token not in stopwords:
                    wv += model.wv[token]
                    l += 1
        if l != 0:
            sents_vec.append(wv/l)
        else:
            sents_vec.append(None)
    return sents_vec

### Textual
datafolder = "./data/"
df = pd.read_csv(os.path.join(datafolder,"t_bbe.csv"))
stopwords = list(set(nltk.corpus.stopwords.words("english")))
model = Word2Vec.load(os.path.join(datafolder, "word2vec_ecc.model"))
ecc = df
documents_raw, documents_token = preprocessW2V(ecc)
sents_vec = vectorize_sent(documents_token, model)

# Images
invlabels_dict = {0:'apple', 1:'orange', 2:'pear'}
resnet18 = models.resnet18(pretrained=True)
for i, param in resnet18.named_parameters():
    param.requires_grad = False
num_ftrs = resnet18.fc.in_features
resnet18.fc = nn.Linear(num_ftrs, 3)
checkpoint = torch.load("./data/model_best.pth.tar", map_location="cpu")
resnet18.load_state_dict(checkpoint['state_dict'])
resnet18.eval()
transform_test = transforms.Compose([transforms.ToTensor(),\
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],\
                                                 std=[0.229, 0.224, 0.225])])
"""
img = img.resize(size=(224,224))
img = np.array(img)
img = transform_test(img)
output = resnet18(img.reshape(1,3,224,224))
_, predicted = torch.max(output.data, 1)
predicted_class = invlabels_dict[predicted.item()]
"""

#We will receive messages that Facebook sends our bot at this endpoint 
@app.route("/", methods=['GET', 'POST'])
def receive_message():
    if request.method == 'GET':
        """Before allowing people to message your bot, Facebook has implemented a verify token
        that confirms all requests that your bot receives came from Facebook.""" 
        token_sent = request.args.get("hub.verify_token")
        return verify_fb_token(token_sent)
    #if the request was not get, it must be POST and we can just proceed with sending a message back to user
    else:
        # get whatever message a user sent the bot
       output = request.get_json()
       for event in output['entry']:
          messaging = event['messaging']
          for message in messaging:
            if message.get('message'):
                #Facebook Messenger ID for user so we know where to send response back to
                recipient_id = message['sender']['id']
                if message['message'].get('text'):
                    #reads user message (usertext)
                    usertext = message['message']['text']
                    send_message(recipient_id, user_query(usertext, model, sents_vec, documents_raw, stopwords))
                    send_message(recipient_id, "Say something and I'll say something related back!!")
                        
                #if user sends us a GIF, photo,video, or any other non-text item
                if message['message'].get('attachments'):
                    response_sent_nontext = get_message()
                    try:
                        attachment_link = message["message"]["attachments"][0]["payload"]["url"]
                        req.urlretrieve(attachment_link, os.path.join("./data", "image_name.jpg"))
                        img = Image.open(os.path.join("./data", "image_name.jpg"))
                        img = img.resize(size=(224,224))
                        img = np.array(img)
                        img = transform_test(img)
                        output = resnet18(img.reshape(1,3,224,224))
                        _, predicted = torch.max(output.data, 1)
                        predicted_class = invlabels_dict[predicted.item()]
                        send_message(recipient_id, f"Oh, its {predicted_class}!!!")
                    except:
                        send_message(recipient_id, response_sent_nontext)
                        pass
    return "Message Processed"


def verify_fb_token(token_sent):
    #take token sent by facebook and verify it matches the verify token you sent
    #if they match, allow the request, else return an error 
    if token_sent == VERIFY_TOKEN:
        return request.args.get("hub.challenge")
    return 'Invalid verification token'


#chooses a random message to send to the user
def get_message():
    sample_responses = ["You are stunning!", "We're proud of you.", "Keep on being you!", "We're greatful to know you :)"]
    # return selected item to the user
    return random.choice(sample_responses)

#uses PyMessenger to send response to user
def send_message(recipient_id, response):
    #sends user the text message provided via input response parameter
    bot.send_text_message(recipient_id, response)
    return "success"

if __name__ == "__main__":
    app.run()