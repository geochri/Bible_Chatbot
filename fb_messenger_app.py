# -*- coding: utf-8 -*-
"""
Created on Tue Sep 11 14:13:09 2018

@author: elelswt
"""
import urllib.request as req
import random
import pickle
from flask import Flask, request
from pymessenger.bot import Bot
from gensim.models import Word2Vec
import nltk
import pandas as pd
import os
from main_bible_chat import BigramChunker, user_query, user_profile, get_name, get_gender, get_age, get_interests, get_verse, get_answers
import torchvision.models as models
import torch.nn as nn
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from waitress import serve
import time

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

def load_pickle(filename):
    completeName = os.path.join("./data/",\
                                filename)
    with open(completeName, 'rb') as pkl_file:
        data = pickle.load(pkl_file)
    return data

### Textual
datafolder = "./data/"
df = pd.read_csv(os.path.join(datafolder,"t_bbe.csv"))
stopwords = list(set(nltk.corpus.stopwords.words("english")))
model = Word2Vec.load(os.path.join(datafolder, "word2vec_ecc.model"))
ecc = df
documents_raw, documents_token = preprocessW2V(ecc)
sents_vec = vectorize_sent(documents_token, model)
bigram_chunker = load_pickle("bigram_chunker.pth.tar")
book_dict = pd.read_csv(os.path.join(datafolder, "key_english.csv"))
book_dict = {book.lower():number for book, number in zip(book_dict["field.1"], book_dict["field"])}

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

### user profiling
users = []
data_path = "./data/profiles/"
for idx,file in enumerate(os.listdir(data_path)):
    filename = os.path.join(data_path,file)
    with open(filename, 'rb') as fo:
        users.append(pickle.load(fo, encoding='bytes'))
start = 0; confirm = 0; name_get = 0; gender_get = 0; age_get = 0; interests_get = 0
bible_mode = 0
user_ids = [u.recipient_id for u in users]
print("Users:", users); print("User_ids:", user_ids)
get_interest_keywords = ["interest","interests","interested","like","love","liking"]
any_interesting = ["interesting", "fun", "nice", "bored", "something", "up"]

#We will receive messages that Facebook sends our bot at this endpoint 
@app.route("/", methods=['GET', 'POST'])
def receive_message():
    global start, name_get, gender_get, age_get, confirm, bible_mode, interests_get
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
                # create new user if doesn't exist else assigns a user profile
                if recipient_id not in user_ids:
                    user = user_profile(); user.recipient_id = recipient_id; user.save(); users.append(user); user_ids.append(recipient_id)
                else:
                    user = users[user_ids.index(recipient_id)]
                
                if message['message'].get('text'):
                    #reads user message (usertext)
                    usertext = message['message']['text']
                    ref = nltk.word_tokenize(usertext.lower())
                    ## create name for user
                    user.name = "friend"; user.save()
                    '''
                    ############################ NAME ############################################################################
                    #### prompts user for name if its a new user
                    if (user.name == "") and (name_get == 0):
                        name_get = 1
                        send_message(recipient_id, "Hey, I have not met you before. I'd like to know more about you!")
                        send_message(recipient_id, "So, how do I address you?")
                        continue
                    #### gets user's name
                    if name_get == 1:
                        name = get_name(usertext, bigram_chunker)
                        if name == None and confirm != 1:
                            send_message(recipient_id, "Ehh sorry I didn't really catch that. Whats your real name again?")
                            continue
                        elif confirm == 0:
                            confirm = 1
                            user.name = name
                            send_message(recipient_id, "Great, can I confirm that %s is your correct name? (yes/no)" % user.name)
                            continue
                        if confirm == 1:
                            if usertext.lower() == "yes":
                                confirm = 0; name_get = 0
                                user.save()
                                send_message(recipient_id, "Great, hi %s!" % user.name)
                            else:
                                confirm = 0
                                send_message(recipient_id, "Okay... so whats your name again?")
                                continue
                    '''
                    '''
                    ################################# GENDER ######################################################################
                    #### prompts user for gender if new user
                    if user.gender == "" and gender_get == 0:
                        gender_get = 1
                        send_message(recipient_id, "Alright %s, so whats your gender?" % user.name)
                        continue
                    ##### gets user's gender
                    if gender_get == 1:
                        gender = get_gender(usertext)
                        if gender == None and confirm != 1:
                            send_message(recipient_id, "Ehh sorry I didn't really catch that. Whats your real gender again?")
                            continue
                        elif confirm == 0:
                            confirm = 1
                            user.gender = gender
                            send_message(recipient_id, "Great, can I confirm that %s you are a %s? (yes/no)" %(user.name, gender))
                            continue
                        if confirm ==1:
                            if usertext.lower() == "yes":
                                confirm = 0; gender_get = 0
                                user.save()
                                send_message(recipient_id, "Great %s!" % user.name)
                            else:
                                confirm = 0
                                send_message(recipient_id, "Ah, ok. So whats your gender again")
                                continue
                        
                    ######################### age ##########################################################
                    ###### prompts user for age if new user
                    if user.age == None and age_get == 0:
                        age_get = 1
                        send_message(recipient_id, "Alright %s, so whats your age :)?" % user.name)
                        continue
                    ##### gets user's age
                    if age_get == 1:
                        age = get_age(usertext)
                        if age == None and confirm != 1:
                            send_message(recipient_id, "Ehh sorry I didn't really catch that. Whats your real age again?")
                            continue
                        elif confirm == 0:
                            confirm = 1
                            user.age = age
                            send_message(recipient_id, "Great, can I confirm that %s you are %s? (yes/no)" %(user.name, age))
                            continue
                        if confirm ==1:
                            if usertext.lower() == "yes":
                                confirm = 0; age_get = 0
                                user.save()
                                send_message(recipient_id, "Great %s!!" % user.name)
                            else:
                                confirm = 0
                                send_message(recipient_id, "Ah, ok why so secretive. So whats your age again???")
                                continue
                    '''
                    '''
                    ############################ interests #######################################################
                    ### prompts user for interests if new user
                    if user.annoyance < 3:
                        if user.interests == [] and interests_get == 0:
                            interests_get = 1
                            send_message(recipient_id, "So %s! What are your interests??" % user.name)
                            continue
                        ###### gets user's interests
                        if interests_get == 1:
                            interests = get_interests(usertext, bigram_chunker)
                            if interests == None and confirm != 1:
                                send_message(recipient_id, "Ehh sorry I didn't really catch that. Whats your interests again?")
                                user.annoyance += 1; user.save()
                                continue
                            elif confirm == 0:
                                confirm = 1
                                user.interests = interests
                                send_message(recipient_id, "Great, can I confirm that %s your interests are " % user.name + ", ".join(interests) \
                                             + "? (yes/no)")
                                continue
                            if confirm ==1:
                                if usertext.lower() == "yes":
                                    confirm = 0; interests_get = 0
                                    user.save()
                                    send_message(recipient_id, "Great %s, interesting!!" % user.name)
                                    start = 1
                                else:
                                    confirm = 0
                                    send_message(recipient_id, "Ah, ok why so secretive. Come on, tell me your interests! :)")
                                    user.annoyance += 1; user.save()
                                    continue
                    '''
                    ############################### Greetings ######################################################
                    if start == 0:
                        start = 1
                        send_message(recipient_id, "Hey, %s. Nice that you're back. What do you wanna know about the Bible?" % user.name)
                        continue
                    bible_mode = 1
                    ######### gets and remember new interests during conversation ####################
                    if any(w for w in ref if w in get_interest_keywords):
                        interests = get_interests(usertext, bigram_chunker)
                        if interests != None:
                            send_message(recipient_id, "I see that you are interested in " + ", ".join(interests) + ".")
                            for interest in interests:
                                if interest not in user.interests:
                                    user.interests.append(interest)
                            user.save(); continue

                    ######## tell user what their current stored interests are #############################
                    if any(w for w in ref if w in ["my", "mine","?","what","whats"]) and \
                        any(w for w in ref if w in ["interests","hobbies","likes",\
                                           "favourites","like","wants"]):
                            send_message(recipient_id, "Oh..")
                            send_message(recipient_id, "Well, looks like you like " + ", ".join(user.interests) + ".")
                            continue
                    
                    ####### go into bible mode #################
                    if bible_mode == 1:
                        book, chapter, verse, verse_end = get_verse(usertext, book_dict)
                        # recommend something interesting to user if user prompts
                        if any(w for w in ref if w in ["what","whats","what's","anything"])\
                            and any(w for w in ref if w in any_interesting):
                            send_message(recipient_id, "Well, since you're interested in " + ", ".join(user.interests) +\
                                         ".....")
                            send_message(recipient_id, user_query(" ".join(user.interests), \
                                                                  model, sents_vec, documents_raw, stopwords))
                            continue
                        # return verse queries
                        elif (book != None) and (chapter != None):
                            if (verse != None) and (verse_end == None): # specific verse
                                send_message(recipient_id, df[(df["b"] == int(book)) & (df["c"] == int(chapter)) & \
                                                                  (df["v"] == int(verse))]["t"].item())
                                continue
                            elif (verse != None) and (verse_end != None): # range of verses
                                send_message(recipient_id, "\n\n".join(df[(df["b"] == int(book)) & (df["c"] == int(chapter)) & \
                                                                  (df["v"] == c)]["t"].item() for c in \
                                                                    range(int(verse)+1, int(verse_end+1))))
                                continue
                            else: # whole chapter
                                send_message(recipient_id, "\n\n".join(df[(df["b"] == int(book)) & (df["c"] == int(chapter)) & \
                                                                  (df["v"] == c)]["t"].item() for c in \
                                                                    range(1, len(df[(df["b"] == int(book)) & (df["c"] == int(chapter))])+1)))
                                continue
                    
                        # Cheers user up if bot detects user is down
                        elif any(w for w in ref if w in ["i", "me", "am"]) and any(w for w in ref if w in ["sad", "suicide", "die",\
                                "lonely", "suicidal", "alone", "disappointed", "heartbroken", "bitter", "distressed",\
                                "down", "gloomy", "low", "troubled", "grief", "dejected", "despondent", "doleful", "hurt"]):
                            send_message(recipient_id, np.random.choice(["Don't worry. Jesus loves you.", "Cheer up! Rejoice in the fact that God us with us",\
                                                                         "Theres nothing to worry if God is with you."]))
                            send_message(recipient_id, user_query(np.random.choice(["peace", "happy", "joy", "loving", "blessing"]), \
                                                                  model, sents_vec, documents_raw, stopwords))
                            user.sad += 1; user.save()
                            continue
                        # searches gotquestions.org and returns answer if user asks a question
                        elif any(w for w in ref if w in ["who", "what", "why", "when", "how", "does", "do", "?"]):
                            send_message(recipient_id, "Please wait while I think...")
                            answer = get_answers(usertext)
                            if answer == None:
                                send_message(recipient_id, user_query(usertext, model, sents_vec, documents_raw, stopwords))
                                send_message(recipient_id, "What do you wanna know about the Bible? :)")
                                continue
                            else:
                                for ans in answer:
                                    send_message(recipient_id, ans); #print(ans)
                                send_message(recipient_id, "What else do you wanna know about the Bible? :)")
                                continue
                        # return similar verses mode
                        else:
                            send_message(recipient_id, user_query(usertext, model, sents_vec, documents_raw, stopwords))
                            send_message(recipient_id, "What do you wanna know about the Bible? :)")
                            continue
                        
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
                        send_message(recipient_id, "Oh, its %s!!!" % predicted_class)
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
    #app.run()
    serve(app, port=5000)