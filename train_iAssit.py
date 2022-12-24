#including necessary libraries
import nltk
import json
import pickle
import numpy as np
import random

#including specific functions from different libraries for building and training the model
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

lst_words=[]
lst_classes = []
lst_documents = []
neglect_words = ['?', '!']
doc_file=open('C:\\Users\\HP\\Desktop\\Colleges\\Mini project\\III sem\\intents.txt', encoding='utf-8').read()
intents= json.loads(doc_file)

for int in intents['intents']:
    for pat in int['patterns']:
        #returns a list of words by tokenizing
        word_tokens=nltk.word_tokenize(pat)
        lst_words.extend(word_tokens)
        #returns the list of words along with the tags
        lst_documents.append((word_tokens, int['tag']))
        #adds distinctive tags to the list of classes
        if int['tag'] not in lst_classes:
            lst_classes.append(int['tag'])
            
#converting the tokenzised words into lower case and lemmatizing them 
lst_words=[lemma.lemmatize(i.lower()) for i in lst_words if i not in neglect_words]
#sorts the list of words and classes and removes duplicates
lst_words=sorted(list(set(lst_words)))
lst_classes=sorted(list(set(lst_classes)))
#documents gives relation between tokenized words and tags
print(len(lst_documents), "DOCUMENTS")
#classes refer to unique tags
print(len(lst_classes), "CLASSES", lst_classes)
#words refer to basically english vocabulary or dictionary
print(len(lst_words), "DISTINCTIVE LEMMATIZED WORDS", lst_words)

#converts words into binary format and stores in a pickle file
pickle.dump(lst_words,open('words_doc.pkl','wb'))
pickle.dump(lst_classes,open('classes_doc.pkl','wb'))

#training data is created
t=[]
#an empty array is created for storing the output
#initialises the array with 0's equal to the length of the classes
op_array=[0] * len(lst_classes)
#generating a bag of words on each sentence present in the training set
for d in lst_documents:
    #initialisation of bag of words
    b=[]
    #considering the list of tokenised words obtained from patterns
    p_words=d[0]
    #creating a root word using lemmitisation,inorder to represent the related words as a single word
    p_words=[lemma.lemmatize(w.lower()) for w in p_words]
    #creating our bag of words by comparing the tokenized words and the lemma words,if the word is found return 1 else return 0
    for j in lst_words:
        b.append(1) if j in p_words else b.append(0)
    #creating our bag of words with array 1
    #mark the index of class that the current pattern is associated to with '1'
    op_row=list(op_array)
    op_row[lst_classes.index(d[1])] = 1
    #add the bag of words and associated classes to training
    t.append([b,op_row])
#re-organising our features that is shuffling our data
random.shuffle(t)
#converting the list into array using np.array
t=np.array(t,dtype=object)
#split the features and target labels a-patterns,b-tags
train_a=list(t[:,0])
train_b=list(t[:,1])
print("Training data created")


#building a neural network model
#defining some paramaters and creating the deep learning model
mod=Sequential()
mod.add(Dense(128,input_shape=(len(train_a[0]),),activation="relu"))
mod.add(Dropout(0.5))
mod.add(Dense(64,activation="relu"))
mod.add(Dropout(0.5))
mod.add(Dense(len(train_b[0]),activation="softmax"))
stgrds=SGD(learning_rate=0.01,decay=1e-6,momentum=0.9,nesterov=True)
mod.compile(loss="categorical_crossentropy",optimizer=stgrds,metrics=["accuracy"]) 
history=mod.fit(np.array(train_a),np.array(train_b),epochs=200,batch_size=5,verbose=1)
mod.save("chatbot_model.h5",history)

print("iAssist model created")
