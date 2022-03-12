#!/usr/bin/env python
# coding: utf-8

# In[46]:


from tkinter import *
from tkinter import messagebox
from tkinter import filedialog
import string
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def clean_text(msg):
    sw=stopwords.words('english')
    sw.remove('not')
    sw.remove("don't")
    sw.remove("didn't")
    sw.remove("hasn't")
    sw.remove("haven't")
    sw.remove("wasn't")
    sw.remove("weren't")
    
    def remove_punct(msg):                               # even this function not required, CountVectorizer can handle this
        return re.sub(f'[{string.punctuation}]','',msg)
    
    def remove_sw_stem(msg):
        ps=PorterStemmer()
        words=word_tokenize(msg)
        new_words=[]
        for w in words:
            if(w not in sw):
                new_words.append(ps.stem(w))
        return " ".join(new_words)
    
    X1=remove_punct(msg)
    X2=X1.lower()                                       # lowercase=True ..parameter can also be passed in CountVectorizer
    X3=remove_sw_stem(X2)
    return X3

'''
    def remove_stopwds(msg):
        words=word_tokenize(msg)
        new_words=[]
        for w in words:
            if(w not in sp_words):
                new_words.append(w)
        return " ".join(new_words)

    def stemming(msg):
        ps=PorterStemmer()
        words=word_tokenize(msg)
        new_words=[]
        for w in words:
            new_words.append(ps.stem(w))
        return " ".join(new_words)
        
    X1=remove_punct(msg)
    X2=X1.lower()
    X3=remove_stopwds(X2)
    X4=stemming(X3)
    return X4
'''

df=pd.read_csv("D:/Ducat/Prac1/Rest. Review App/Restaurant_Reviews.txt",delimiter='\t')
df.Review=list(map(clean_text,df.Review))
cv=CountVectorizer(binary=False,ngram_range=(1,2))
X=cv.fit_transform(df.Review).toarray()
y=df.Liked
model=MultinomialNB()
model.fit(X,y)

win=Tk()
win.state('zoomed')
win.resizable(width=False,height=False)
win.configure(bg='yellow')
win.title('Restaurant Review Analysis')

lbl_title=Label(win,text='Review Analysis',font=('Arial',50,'bold','underline'),bg='yellow')
lbl_title.place(relx=.3,rely=0)

def home_screen():
    frm=Frame(win,bg='#2a9d8f')
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
    
    global entry_user,entry_pass
    
    lbl_user=Label(frm,text='User Name',font=('Arial',20,'bold'),bg='#2a9d8f')
    lbl_user.place(relx=.28,rely=.3)
    
    entry_user=Entry(frm,font=('Arial',20,'bold'),bd=10)
    entry_user.place(relx=.42,rely=.3)
    entry_user.focus()
    
    lbl_pass=Label(frm,text='Password',font=('Arial',20,'bold'),bg='#2a9d8f')
    lbl_pass.place(relx=.28,rely=.4)
    
    entry_pass=Entry(frm,font=('Arial',20,'bold'),bd=10,show='*')
    entry_pass.place(relx=.42,rely=.4)
    
    btn_login=Button(frm,text='Login',font=('Arial',20,'bold'),bd=10,width=10,command=lambda:welcome_screen())
    btn_login.place(relx=.45,rely=.5)
    
    lbl_copyright=Label(win,text='© Abhinav Gupta',font=('Arial',20,'bold'),bg='#2a9d8f')
    lbl_copyright.place(relx=.82,rely=.90)
    
def welcome_screen():
    user=entry_user.get()
    pwd=entry_pass.get()
    if(len(user)==0 or len(pwd)==0):
        messagebox.showerror('Validation',"Username/Password can't be empty")
    else:
        if(user.lower()=='admin' and pwd.lower()=='abhinav'):
            frm=Frame(win,bg='#2a9d8f')
            frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
            
            btn_single=Button(frm,text='Single Feedback Prediction',font=('Arial',20,'bold'),bd=10,width=25,command=lambda:single_feedback_screen())
            btn_single.place(relx=.3,rely=.2)
                      
            btn_bulk=Button(frm,text='Bulk Feedback Prediction',font=('Arial',20,'bold'),bd=10,width=25,command=lambda:bulk_feedback_screen())
            btn_bulk.place(relx=.3,rely=.4)
                      
            btn_logout=Button(frm,text='Logout',font=('Arial',20,'bold'),bd=5,width=10,command=lambda:logout())
            btn_logout.place(relx=.4,rely=.6)
            
            lbl_copyright=Label(win,text='© Abhinav Gupta',font=('Arial',20,'bold'),bg='#2a9d8f')
            lbl_copyright.place(relx=.82,rely=.90)
            
        else:
            messagebox.showerror('Login','Invalid Username/Password')
            
def logout():
    option=messagebox.askyesno('Confirmation','Do you want to logout')
    if(option==True):
        home_screen()
    else:
        pass

def single_feedback_screen():
    frm=Frame(win,bg="#2a9d8f")
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
    
    lbl_user=Label(frm,text="Enter Feedback :",font=('',20,'bold'),bg="#2a9d8f")
    lbl_user.place(relx=.1,rely=.25)
    
    entry_user=Entry(frm,font=('',20,'bold'),bd=10,width=50)
    entry_user.place(relx=.3,rely=.24)
    entry_user.focus()
    
    btn_predict=Button(frm,text="Predict",font=('',20,'bold'),bd=10,width=10,command=lambda:predict_single(entry_user,lbl_result))
    btn_predict.place(relx=.32,rely=.5)
    
    lbl_result=Label(frm,text='Prediction',font=('',20,'bold'),bg='gray',fg='yellow')
    lbl_result.place(relx=.63,rely=.53)
    
    btn_back=Button(frm,command=lambda:welcome_screen(),text="Back",font=('',20,'bold'),bd=10)
    btn_back.place(relx=.9,rely=0)
            
    lbl_copyright=Label(win,text='© Abhinav Gupta',font=('Arial',20,'bold'),bg='#2a9d8f')
    lbl_copyright.place(relx=.82,rely=.90)
    
def predict_single(entry_user,lbl_result):
    user_review=entry_user.get()
    ct=clean_text(user_review)
    X_test=cv.transform([ct]).toarray()
    pred=model.predict(X_test)
    if(pred[0]==0):
        lbl_result.configure(text='Not Liked 👎',fg='red')
    else:
        lbl_result.configure(text='Liked 👍',fg='white')
        
def bulk_feedback_screen():
    frm=Frame(win,bg="#2a9d8f")
    frm.place(relx=0,rely=.15,relwidth=1,relheight=1)
    
    lbl_src=Label(frm,text='Select Source File:',font=('',20,'bold'),bg='#2a9d8f')
    lbl_src.place(relx=.03,rely=.21)
    
    entry_src=Entry(frm,font=('',20,'bold'),bd=10,width=40)
    entry_src.place(relx=.3,rely=.2)
    entry_src.focus()
    
    lbl_dest=Label(frm,text='Select Destination Folder:',font=('',20,'bold'),bg='#2a9d8f')
    lbl_dest.place(relx=.03,rely=.41)
    
    entry_dest=Entry(frm,font=('',20,'bold'),bd=10,width=40)
    entry_dest.place(relx=.3,rely=.4)
    
    btn_browse1=Button(frm,text='Browse',font=('',20,'bold'),bd=10,command=lambda:browse1(entry_src))
    btn_browse1.place(relx=.79,rely=.19)
    
    btn_browse2=Button(frm,text='Browse',font=('',20,'bold'),bd=10,command=lambda:browse2(entry_dest))
    btn_browse2.place(relx=.79,rely=.39)
    
    btn_predict=Button(frm,text='Predict & Save',font=('',20,'bold'),bd=10,command=lambda:predict_save(entry_src,entry_dest))
    btn_predict.place(relx=.4,rely=.6)
    
    btn_back=Button(frm,command=lambda:welcome_screen(),text="Back",font=('',20,'bold'),bd=10)
    btn_back.place(relx=.9,rely=0)
            
    lbl_copyright=Label(win,text='© Abhinav Gupta',font=('Arial',20,'bold'),bg='#2a9d8f')
    lbl_copyright.place(relx=.82,rely=.90)
    
def browse1(entry_path):
    file_path=filedialog.askopenfilename()
    entry_path.delete(0,'end')                        # or END
    entry_path.insert(0,file_path)
    
def browse2(dest_path):
    file_path=filedialog.askdirectory()+"/result.txt"
    dest_path.delete(0,END)
    dest_path.insert(0,file_path)
    
def predict_save(entry_src,entry_dest):
    srcpath=entry_src.get()
    destpath=entry_dest.get()
    df=pd.read_csv(srcpath,names=['Review'])
    X=df.Review.map(clean_text)                       # or,    df.Review=list(map(clean_text,df.Review))
    X_test=cv.transform(X).toarray()                  # or,    X_test=cv.transform(df.Review).toarray()
    pred=model.predict(X_test)
    result_df=pd.DataFrame()
    result_df['Review']=df.Review
    result_df['Sentiment']=pred
    result_df['Sentiment']=result_df['Sentiment'].map({0:'Not Liked',1:'Liked'})
    result_df.to_csv(destpath,index=False,sep='\t')
    messagebox.showinfo('Result','Prediction Saved')
    
home_screen()
win.mainloop()


# In[ ]:




