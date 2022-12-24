#importing the necessary packages
import nltk
import pickle
import numpy as np
from nltk.stem import WordNetLemmatizer
lemma=WordNetLemmatizer()

#loads the saved model
from keras.models import load_model
mod=load_model("C:\\Users\\HP\\Desktop\\Colleges\\Mini project\\III sem\\chatbot_model.h5")
import tkinter
from tkinter import *


#load the json file and pickle files created during our training the model
import json
import random
intents=json.loads(open('C:\\Users\\HP\\Desktop\\Colleges\\Mini project\\III sem\\intents.txt',encoding='utf-8').read())
lst_words=pickle.load(open('C:\\Users\\HP\\Desktop\\Colleges\\Mini project\\III sem\\words_doc.pkl','rb'))
lst_classes=pickle.load(open('C:\\Users\\HP\\Desktop\\Colleges\\Mini project\\III sem\\classes_doc.pkl','rb'))

#implementing functions to retrieve us a random response from list of responses
def clear_up_text(line):
    # tokenisation of pattern that is user input
    line_words=nltk.word_tokenize(line)
    #stemming each word that is creating a short form
    line_words=[lemma.lemmatize(i.lower()) for i in line_words]
    return line_words

def bag_of_words(line, lst_words,display_details=True):
    #calling the function
    line_words=clear_up_text(line)
    #returns bag of words
    a=[0]*len(lst_words)  
    b=[0]*len(lst_words)  
    for l in line_words:
        for c,d in enumerate(lst_words):
            if d==l: 
                #assign 1 if each word in the bag exists in the sentence
                b[c]=1
                if display_details:
                    print("got in bag:%l" % d)
    return(np.array(b))

def guess_class(line, mod):
    g=bag_of_words(line, lst_words,display_details=False)
    result=mod.predict(np.array([g]))[0]
    threshold=0.25
    pred=[[y,z] for y,z in enumerate(result) if z>threshold]
    pred.sort(key=lambda x: x[1], reverse=True)
    list=[]
    for p in pred:
        list.append({"int": lst_classes[p[0]], "probability": str(p[1])})
    return list

def getresponse(intents_list, intents_json):
    tag=intents_list[0]['int']
    listofintents=intents_json['intents']
    for l in listofintents:
        if(l['tag']==tag):
            ans=random.choice(l['responses'])
            break
    return ans

def iAssist_response(message):
    intent=guess_class(message, mod)
    answer=getresponse(intent, intents)
    return answer

#Create Graphical user interface

count=0
def send():
    global count
    count+=1

    if(count==1):
        TextPanel.config(state=NORMAL)
        TextPanel.config(foreground="#442265",font=("Verdana",12))
        TextPanel.insert(END,"Hi, my name is iAssist\nI can give you general information about New Horizon\nCollege of Engineering\n")
        TextPanel.insert(END,"I can tell you about:\n")
        TextPanel.insert(END, "1. Principal, Managing directories and location of the college\n")
        TextPanel.insert(END, "2. Courses, duration of the course and eligibility criteria\n")
        TextPanel.insert(END,"3. Exams, Internships and scholarship\n4. Bus facility, timings of the college\n")
        TextPanel.insert(END,"5. Website of the college\n6. Basic facilities provided by the college\n")
        TextPanel.insert(END,"7. Hostel, training and placements\n")
        TextPanel.insert(END,"8. Student Clubs,labs and cafe\n9. Library\n\n")
        TextPanel.insert(END,"I can assist you through any of these and give detailed\ninformation about them\n")
        TextPanel.insert(END,"Please type your query\n\n")
        TextPanel.config(state=DISABLED)
        TextPanel.yview(END)
    
    sms=EntryPanel.get("1.0",'end-1c').strip()
    EntryPanel.delete("0.0",END)
    
    if sms != '':
        TextPanel.config(state=NORMAL)
        TextPanel.insert(END, "student: " + sms + '\n\n')
        TextPanel.config(foreground="#442265", font=("Verdana", 12 ))
    
        ans = iAssist_response(sms)
        TextPanel.insert(END, "iAssist: " + ans + '\n\n')
            
        TextPanel.config(state=DISABLED)
        TextPanel.yview(END)
    else:
        TextPanel.config(state=NORMAL)
        TextPanel.insert(END, "student: " + sms + '\n\n')
        TextPanel.config(foreground="#442265", font=("Verdana", 12 ))
        TextPanel.insert(END, "iAssist: " + "sorry didnt get you" + '\n\n')
        TextPanel.config(state=DISABLED)
        TextPanel.yview(END)
            
EntryPage=Tk()
EntryPage.title("iAssist-student queries")
EntryPage.geometry("550x700")
EntryPage.resizable(width=FALSE,height=FALSE)

#Creating user interface
TextPanel=Text(EntryPage,bd=0,bg="ivory",height="8",width="50",font="Centaur")

TextPanel.config(state=DISABLED)

#Set specifications of scroll bar and attach it to the interface
scroller=Scrollbar(EntryPage,command=TextPanel.yview,cursor="heart")
TextPanel['yscrollcommand']=scroller.set

#Creating a button to send questions from the user
sendicon=Button(EntryPage,font=("Garamond",16,'bold'),text="Enter",width="12",height=5,bd=0,bg="red",activebackground="white",fg='white',command= send )

#Creating a textbox to enter questions
EntryPanel=Text(EntryPage,bd=0,bg="#23e8ad",width="29",height="5",font="Arial")
#EntryPanel.bind("<Return>", send)


#Attaching all the specifications(Scroll bar,send,entrybox)on the interface
scroller.place(x=520,y=6,height=556)
TextPanel.place(x=6,y=6,height=556,width=509)
EntryPanel.place(x=150,y=568,height=127,width=366)
sendicon.place(x=6,y=568,height=127)

EntryPanel.mainloop()
