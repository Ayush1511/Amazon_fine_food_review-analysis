#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[10]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pickle


# In[3]:


df=pd.read_csv('Reviews.csv')
df.head()


# In[4]:


df.Score.value_counts()


# In[5]:


df.drop(['Id','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator','Time','Summary','ProductId'],axis=1,inplace=True)
df.head()


# In[6]:


df=df.sample(n=10000)
len(df)


# In[7]:


type(df)


# In[8]:


df['Score'].replace(to_replace =[1,2], 
                            value ="Negative",inplace=True)
df['Score'].replace(to_replace =[3], 
                            value ="Neutral",inplace=True)
df['Score'].replace(to_replace =[4,5], 
                            value ="Positive",inplace=True)
df.head()


# In[9]:


df.Score.value_counts()


# In[10]:


df.Text[0:10]


# Preprocessing:-
# 1. Clean
# 2. Tokenize
# 3. Stopwords
# 4. Lower
# 5. Lemmatize/Stemmer

# In[11]:


#Stopwords and Puncs
import string
from nltk.corpus import stopwords
stop = stopwords.words('english')
punctuations = list(string.punctuation)
l_1=['br','http','href','wwe','amazon','com']
stop = stop + punctuations+l_1
stop[0:5]


# In[12]:


X=df.Text.values
Y=df.Score.values


# In[13]:


import nltk
nltk.download('punkt')


# ## Data Cleaning

# In[14]:


#Removes digits and keeps only lower case alphabets
import re
for i in range(len(X)):
    X[i]=X[i].lower()
    X[i]=re.sub("[^a-z]+"," ",X[i])


# ### Tokenization

# In[15]:


# Tokenize
from nltk import word_tokenize
for i in range(len(X)):
    X[i]=word_tokenize(X[i])
#     if i%100:
#         print("...")


# ### Stopwords Removal

# In[47]:


# Remove stopwords
for i in range(len(X)):
    words=[word for word in X[i] if word not in stop and len(word)>=2]
    X[i]=words
X[0]


# ### Lemmatization

# In[48]:


from nltk.corpus import wordnet
def get_simple_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


# In[49]:


from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()


# In[50]:


import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# In[51]:


#word_doc=[]
for i in range(len(X)):
    for j in range(len(X[i])):
        pos=pos_tag([X[i][j]])
        X[i][j]=lemmatizer.lemmatize(X[i][j],pos=get_simple_pos(pos[0][1]))
        #word_doc.append(clean_word.lower())
    X[i]=' '.join(X[i])
    if i%1000==0:
        print(i)


# In[52]:


X[8]


# ### Saving reviews and their class to Text file

# In[53]:


f=open("Cleaned_review","w+")
for i in range(len(X)):
    f.write(X[i])
    f.write('\n')
f.close()


# In[54]:


f=open("Class","w+")
for i in range(len(Y)):
    f.write(Y[i])
    f.write('\n')
f.close()


# 

# In[55]:


#cleaned_text=[]
f=open("Cleaned_review","r+")
text=f.read()
f.close()
text=text.split("\n")
text.pop()
    


# In[56]:


f=open("Class","r+")
ans=f.read()
f.close()
ans=ans.split("\n")
ans.pop()


# In[57]:


len(text)


# In[58]:


len(ans)


# In[59]:


ans[54]


# ### Frequency of every word

# In[60]:


total_words=[]
for i in range(len(text)):
    l=text[i].split()
    for j in range(len(l)):
        total_words.append(l[j])
        


# In[61]:


len(total_words)


# In[62]:


import collections
counter = collections.Counter(total_words)
freq_cnt = dict(counter)
print(len(freq_cnt.keys()))


# ### Selecting words which have count>10

# In[64]:


useful_words=[]
sorted_freq_cnt = sorted(freq_cnt.items(),reverse=True,key=lambda x:x[1])

# Filter
threshold = 10
sorted_freq_cnt  = [x for x in sorted_freq_cnt if x[1]>threshold]
useful_words = [x[0] for x in sorted_freq_cnt]


# In[65]:


print(len(useful_words))


# ## Creating final corpus

# In[66]:


text_final=[]
for i in range(len(text)):
    l_2=text[i].split()
    l_3=[x for x in l_2 if x in useful_words]
    text_final.append(l_3)
    text_final[i]=' '.join(text_final[i])
        


# In[67]:


text_final[9]


# ## Saving Final reviews to text file

# In[68]:


f=open("Cleaned_review_2","w+")
for i in range(len(text_final)):
    f.write(text_final[i])
    f.write('\n')
f.close()


# In[69]:


len(text_final)


# In[ ]:





# ## Analysis

# In[1]:


f=open("Cleaned_review_2","r+")
data=f.read()
f.close()
data=data.split("\n")
data.pop()
    


# In[2]:


data[76]


# In[3]:


len(data)


# In[ ]:





# In[3]:


f=open("Class","r+")
result=f.read()
f.close()
result=result.split("\n")
result.pop()


# In[4]:


type(result)


# In[5]:


result[76]


# In[6]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.30, random_state=42)


# In[7]:


len(y_train)


# In[ ]:





# ## Count Vectorization

# In[34]:


from sklearn.feature_extraction.text import CountVectorizer


# In[49]:


cv = CountVectorizer()
x_vec = cv.fit_transform(x_train)


# In[50]:


pickle.dump(cv, open('tranform.pkl', 'wb'))


# In[51]:


print(x_vec.shape)
print(x_vec[0])


# In[52]:


xtest_vec = cv.transform(x_test)


# In[ ]:





# ### Machine Learning Models

# In[19]:


from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns


# In[53]:


def scoring_analysis(y_train,y_train_pred,y_test,y_test_pred):
    print("Training Score : ",accuracy_score(y_train,y_train_pred))
    print("Testing Score : ",accuracy_score(y_test,y_test_pred))
    labels=['Positive','Negative','Neutral']
    print(sns.heatmap(confusion_matrix(y_test, y_test_pred,labels=labels),annot=True))
    print("f1-Score : ",f1_score(y_test, y_test_pred,average='micro'))


# In[194]:


from sklearn.naive_bayes import MultinomialNB
def naive_bayes(xtrain,xtest,ytrain,ytest):
    mb=MultinomialNB()
    mb.fit(xtrain,ytrain)
    y_train_pred=mb.predict(xtrain)
    y_test_pred=mb.predict(xtest)
    print("Multinomial Naive Bayes :-")
    scoring_analysis(y_train,y_train_pred,y_test,y_test_pred)


# In[54]:


from sklearn.ensemble import RandomForestClassifier
def random_forest(xtrain,xtest,ytrain,ytest,n):
    rf=RandomForestClassifier(n_estimators=n)
    rf.fit(xtrain,ytrain)
    filename = 'nlp_model.pkl'
    pickle.dump(rf, open(filename, 'wb'))
    y_train_pred=rf.predict(xtrain)
    y_test_pred=rf.predict(xtest)
    print("Random Forest Classifier :-")
    scoring_analysis(y_train,y_train_pred,y_test,y_test_pred)


# In[196]:


from sklearn import svm
def support_vector_machine(xtrain,xtest,ytrain,ytest,k):
    svc = svm.SVC(kernel=k)
    svc.fit(xtrain,ytrain)
    y_train_pred=svc.predict(xtrain)
    y_test_pred=svc.predict(xtest)
    print("Support Vector Classifier :-")
    print("Using ",k)
    scoring_analysis(y_train,y_train_pred,y_test,y_test_pred)
    print()


# In[197]:


naive_bayes(x_vec,xtest_vec,y_train,y_test)


# In[55]:


random_forest(x_vec,xtest_vec,y_train,y_test,100)


# In[199]:


kernels=['linear','poly','rbf']
for k in kernels:
    support_vector_machine(x_vec,xtest_vec,y_train,y_test,k)


# 

# ### Best Model - Random forest classifier
# #### F1 Score on testing data - 81.2%

# In[ ]:


## Random forest model saved in pickle file


# In[ ]:





# ### Analysis

# In[156]:


from wordcloud import WordCloud
def wc(data,bgcolor,title,s):
    print(s)
    plt.figure(figsize = (100,100))
    wc = WordCloud(background_color = bgcolor, max_words = 1000,  max_font_size = 50)
    wc.generate(' '.join(data))
    plt.imshow(wc)
    plt.axis('off')


# In[157]:


sns.set(rc={'figure.figsize':(11.7,8.27)})


# In[158]:


def bag_of_words(data):
    bag=[]
    for i in range(len(data)):
        temp=data[i].split()
        for j in range(len(temp)):
            bag.append(temp[j])
    return bag


# In[159]:


from collections import Counter
def word_with_sentiment(data,s):
    bag=bag_of_words(data)
    counter=Counter(bag)
    most=counter.most_common(30)
    x, y= [], []
    for word,count in most[:40]:
        x.append(word)
        y.append(count)
    print(s)
    sns.barplot(x=y,y=x)


# In[160]:


positve=[]
negative=[]
neutral=[]
for i in range(len(data)):
    if result[i]=='Negative':
        negative.append(data[i])
    elif result[i]=='Positive':
        positve.append(data[i])
    else:
        neutral.append(data[i])


# In[161]:


word_with_sentiment(positve,'Positive')


# In[162]:


word_with_sentiment(neutral,'Neutral')


# In[163]:


word_with_sentiment(negative,'Negative')


# In[164]:


wc(positve,'white','Common Words','Positive')


# In[165]:


wc(negative,'white','Common Words','Negative')


# In[166]:


wc(neutral,'white','Common Words','Neutral')


# In[ ]:





# ## Model prediction

# In[56]:


filename = 'nlp_model.pkl'
clf = pickle.load(open(filename, 'rb'))
cv=pickle.load(open('tranform.pkl','rb'))


# In[60]:


message = "good classic movie put halloween geena davis alex baldwin micheal keaton earlier day good job real cute movie"


# In[61]:


data = [message]


# In[62]:


vect = cv.transform(data).toarray()


# In[63]:


my_prediction = clf.predict(vect)


# In[67]:


my_prediction[0]


# In[65]:


print (my_prediction)


# In[68]:



from flask import Flask,render_template,url_for,request

