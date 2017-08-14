# -*- coding: utf-8 -*-
"""
Created on Wed Aug 09 20:16:45 2017

@author: Dana
"""
#======================================================================
#---------------------PRELUDE----------------------
#IMPORT NEEDED MODULES, SET WORKING DIRECTORY, ETC.
import os
os.getcwd()
wd = 'C:\Users\Dana\Desktop\DATA TEXT MINING'
os.chdir(wd)
import glob, io,  operator
import pandas as pd
import textminer as tm
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import matplotlib.pyplot as plt
import quickndirty as qd
from collections import defaultdict
from operator import itemgetter
from gensim import corpora
import plotly
import plotly.plotly as py
from plotly.graph_objs import *
import plotly.graph_objs as go
filepath = os.path.join(wd,'LETTERS\\')
filenames = glob.glob(filepath + '*.txt')
os.chdir(filepath)
#=======================================================================
#--------------------PART I: DOWNLOADING LETTERS------------------------
#GET LIST OF BOOKS
my_url_books = 'http://danmarksbreve.kb.dk/brevudgivelser'
#OPENS CONNECTION, GRABS THE PAGE
client_books = uReq(my_url_books)
page_html_books = client_books.read()
client_books.close()
#PARSES INTO HTML
page_soup_books = soup(page_html_books, "html.parser")
#HTML OF ALL THE BOOKS ON THE WEBSITE         
containers_books = page_soup_books.findAll("div", {"itemtype":"http://schema.org/Thing"}) #grabs all html objects to a list
#ID FOR EVERY BOOK AND APPEND TO A LIST               
books_list=[]
for container in containers_books:
    book = container.a["data-context-href"]
    if len(books_list) < 91:
        books_list.append(book[:-6])
#GET LIST OF LETTERS
#LIST OF ALL URLS FOR THE BOOKS     
my_url_letters = 'http://danmarksbreve.kb.dk'
urls_books=[]
for book in books_list:
    url = my_url_letters + book
    urls_books.append(url)
#MAKE A LIST FOR ALL LETTERS
letters_list=[]                                    
for url in urls_books:                                         
    client = uReq(url)
    page_html = client.read()
    client.close()
    page_soup = soup(page_html, "html.parser")
    containers = page_soup.findAll("a")                                                 
    for container in containers:
        link = container["href"]
        if len(letters_list) < 20000:
            if len(my_url_letters + '/catalog' + link) > 90:
                letters_list.append(my_url_letters + '/catalog' + link[1:] + '.pdf')                                                  
#==============================================================================
#-------------------------PART II: PREPROCESSING-------------------------------
#READ TEXT LINE BY LINE
def read_txt_lines(filepath):
    with io.open(filepath, 'r', encoding = 'utf-8') as f:
        content = f.readlines()
    return content
#IMPORT DATA
content_list = []
for filename in filenames:
    text = read_txt_lines(filename)
    content_list.append(text)
#SEPARATE TEXT FROM METATEXT
text = []
metadata = []
letters_text = []
letters_metadata = []
for letter in content_list:
    text = letter[:-4]
    letters_text.append(text)
    metadata = letter[-4:]
    letters_metadata.append(metadata)
#ALL IN ONE
metadata_unified = []
metadata_unified_list = []
for letter in letters_metadata:
     metadata_unified = ''.join(letter)
     metadata_unified_list.append(metadata_unified)
#MATCH DATE, AUTHOR, ETC.
index = 0
match_date = ''
letters_dates = []
match_author = ''
letters_author = []
match_recipient = ''
letters_recipient = []
for letter in metadata_unified_list:
    match_date = re.compile(r"\d{4}.*[)]").search(letter) 
    if match_date is not None:  
        letters_dates.append(match_date.group()[:-1])
    else: 
        letters_dates.insert(index,'0')   
    match_author = re.compile(r"FRA:(.*?)\d{4}").search(letter)
    if match_author is not None:
        letters_author.append(match_author.group()[5:-6]) 
    else:
        letters_author.insert(index,'noauthor')
    match_recipient = re.compile(r"BREV TIL:(.*)FRA:").search(letter) 
    if match_recipient is not None:
        letters_recipient.append(match_recipient.group()[10:-5])
    else:
        letters_recipient.insert(index,'norecipient')
    index = index + 1
#GET THE YEARS
year = ''
letters_years = []
for text in letters_dates:
        year = text[:4]
        letters_years.append(year)
#TEXT AS A STRING
letters_text_strings = []
string = ''
for text in letters_text:
    string = ''.join(text)
    letters_text_strings.append(string)
#STRING OF METADATA
letters_metadata_strings = []
string = ''
for text in letters_metadata:
    string = ''.join(text)
    letters_metadata_strings.append(string)
#CLEAN THE TEXT A BIT
cleaner = ''
metd = ''
index = 0
letters_text_clean = []
for text in letters_text_strings:
    metd = letters_metadata_strings[index]
    cleaner = text.replace(metd,'')
    letters_text_clean.append(cleaner)
    index = index + 1
#==========================================================================
#TAKE OUT NUMBERS
letters_clean=[]
for letter in letters_text_clean:
      sub =  re.sub(r'\d','',letter.lower())
      letters_clean.append(sub)
#TOKENIZER  
def tokenize(s, n = 1, lentoken = 0):
   if type(s) == unicode:
       tokenizer = re.compile(r'\W*', re.UNICODE)
   else:
       tokenizer = re.compile(r'\W*')
   unigram = tokenizer.split(s)
   unigram = [token.lower() for token in unigram if len(token) > lentoken] 
   if n > 1:
       return [unigram[i:i+n] for i in range(len(unigram)-(n-1))]
   else:
       return [s for s in unigram if s]
#LIST OF TOKENS, MORE THAN 3 CHARACTERS
tokens = ''
letters_tokens = [] 
for text in letters_text_clean:
    tokens = tokenize(text,1,lentoken = 3)
    letters_tokens.append(tokens)
#=======================================================================
#PUT INTO DATAFRAME
df = pd.DataFrame()
df["text"] = letters_text_clean
df["metadata"] = letters_metadata
df["date"] = letters_dates
df["year"] = letters_years
df["author"] = letters_author
df["recipient"] = letters_recipient
df["tokens"] = letters_tokens
#=========================================================================
#PLOT LETTER DENSITY FOR EACH YEAR THERE WERE LETTERS
lexicon = set(letters_years)
yf_all = dict([(year, letters_years.count(year)) for year in lexicon])
yf_sorted = sorted(yf_all.items())
list_years = []
list_freq = []
year = ''
freq = ''
for string in yf_sorted:
   year = string[0]
   freq = string[1]
   list_years.append(year)
   list_freq.append(freq)   
qd.plotvars(list_years[1:],list_freq[1:],sv=True)
#===========================================================================
#FREQUENCY OF ALL TOKENS
t_f_total = defaultdict(int) 
for letter in letters_tokens:
    for token in letter:
        t_f_total[token] += 1 
#PUT IN DICTIONARY AND SORT
tf = dict(t_f_total)
tf_sort = sorted(tf.items(), key = operator.itemgetter(1), reverse = True)   
#============================================================================
#CATEGORIZE BY CENTURY
period = []
index = 0
for i in df["year"]:
    if i >= "1500" and i < "1600":
        period.insert(index, "1500")
    elif i >= "1600" and i < "1700":
        period.insert(index, "1600")
    elif i >= "1700" and i < "1800":
        period.insert(index, "1700")
    elif i >= "1800":
        period.insert(index, "1800")
    else:
        period.insert(index,"0")
    index = index + 1
df["period"] = period
#NEW LIST OF TOKENS SORTED BY CENTURY
six_cen = []
sev_cen = []
eig_cen = []
nin_cen = []
non_cen = [] 
index = 0
for token in letters_tokens:
    if period[index] == "1500":
        six_cen.append(token)
    elif period[index] == "1600":
        sev_cen.append(token)
    elif period[index] == "1700":
        eig_cen.append(token)
    elif period[index] == "1800":
        nin_cen.append(token)
    else:
        non_cen.append(token)
    index = index + 1  
#=============================================================================
#FLAT LIST OF TOKENS PER CENTURY
n_6 = []
for letter in six_cen:
    for token in letter:
        n_6.append(token)
n_7 = []
for letter in sev_cen:
    for token in letter:
        n_7.append(token)     
n_8 = []
for letter in eig_cen:
    for token in letter:
        n_8.append(token)       
n_9 = []
for letter in nin_cen:
    for token in letter:
        n_9.append(token)  
#========================================================================
#GET WORD STEMS FOR EACH CENTURY
stemmer = SnowballStemmer("danish")
nn_6 = []
for token in n_6:
    n = stemmer.stem(token)
    nn_6.append(n)
nn_7 = []
for token in n_7:
    n = stemmer.stem(token)
    nn_7.append(n)
nn_8 = []
for token in n_8:
    n = stemmer.stem(token)
    nn_8.append(n)
nn_9 = []
for token in n_9:
    n = stemmer.stem(token)
    nn_9.append(n)
#============================================================================
#-------------------------PART III: ANALYSIS-------------------------------
#LOOK AT INDIVIDUAL WORD FREQUENCIES
def tf(term, tokens):
       result = tokens.count(term)
       return result
#~FREQUENCY OF WORD DIVIDED BY NUMBER OF WORDS IN LIST, AS PERCENTAGE~
#SIXTEENTH CENTURY
six_danmark =  tf('danmark', nn_6)/float(len(nn_6))*100         #Denmark
six_krig =  tf('krig', nn_6)/float(len(nn_6))*100               #war
six_peng =  tf('peng', nn_6)/float(len(nn_6))*100               #money
six_svensk =  tf('svensk', nn_6)/float(len(nn_6))*100           #Sweedish
six_norsk =  tf('norsk', nn_6)/float(len(nn_6))*100             #Norwegian
six_udenlandsk =  tf('udenlandsk', nn_6)/float(len(nn_6))*100   #foreigner
six_lov =  tf('lov', nn_6)/float(len(nn_6))*100                 #lov
six_fred =  tf('fred', nn_6)/float(len(nn_6))*100               #frederik
six_christ =  tf('christ', nn_6)/float(len(nn_6))*100           #christian
six_folk =  tf('folk', nn_6)/float(len(nn_6))*100               #people    
#SEVENTEENTH CENTURY
sev_danmark = tf('danmark', nn_7)/float(len(nn_7))*100
sev_krig = tf('krig', nn_7)/float(len(nn_7))*100
sev_peng = tf('peng', nn_7)/float(len(nn_7))*100
sev_svensk = tf('svensk', nn_7)/float(len(nn_7))*100
sev_norsk = tf('norsk', nn_7)/float(len(nn_7))*100
sev_udenlandsk = tf('udenlandsk', nn_7)/float(len(nn_7))*100
sev_udlejeren =  tf('udlejeren', nn_7)/float(len(nn_7))*100     
sev_lov =  tf('lov', nn_7)/float(len(nn_7))*100                 
sev_fred =  tf('fred', nn_7)/float(len(nn_7))*100           
sev_christ =  tf('christ', nn_7)/float(len(nn_7))*100        
sev_folk =  tf('folk', nn_7)/float(len(nn_7))*100               
#EIGHTEENTH CENTURY
eig_danmark = tf('danmark', nn_8)/float(len(nn_8))*100
eig_krig = tf('krig', nn_8)/float(len(nn_8))*100
eig_peng = tf('peng', nn_8)/float(len(nn_8))*100
eig_svensk = tf('svensk', nn_8)/float(len(nn_8))*100
eig_norsk = tf('norsk', nn_8)/float(len(nn_8))*100
eig_udenlandsk = tf('udenlandsk', nn_8)/float(len(nn_8))*100
eig_lov =  tf('lov', nn_8)/float(len(nn_8))*100                 
eig_fred =  tf('fred', nn_8)/float(len(nn_8))*100               
eig_christ =  tf('christ', nn_8)/float(len(nn_8))*100     
eig_folk =  tf('folk', nn_8)/float(len(nn_8))*100               
#NINETEENTH CENTURY
nin_danmark = tf('danmark', nn_9)/float(len(nn_9))*100
nin_krig = tf('krig', nn_9)/float(len(nn_9))*100
nin_peng = tf('peng', nn_9)/float(len(nn_9))*100
nin_svensk = tf('svensk', nn_9)/float(len(nn_9))*100
nin_norsk = tf('norsk', nn_9)/float(len(nn_9))*100
nin_udenlandsk = tf('udenlandsk', nn_9)/float(len(nn_9))*100
nin_lov =  tf('lov', nn_9)/float(len(nn_9))*100                 
nin_fred =  tf('fred', nn_9)/float(len(nn_9))*100               
nin_christ =  tf('christ', nn_9)/float(len(nn_9))*100           
nin_folk =  tf('folk', nn_9)/float(len(nn_9))*100  
#============================================================================
#~FANCY BAR PLOT SHOWING FREQUENCY OF WORDS PER CENTURY~
#USE THIS LINE TO GET LINK TO INTERACTIVE GRAPH HTML!
plotly.tools.set_credentials_file(username='au558796', api_key='SiqFzA4VCVxfHfOyldj3')
six = go.Bar(
    x=['danmark', 'krig', 'peng', 'svensk', 'norsk', 'udenlandsk','lov','fred','christ','folk'],
    y=[six_danmark, six_krig, six_peng, six_svensk, six_norsk, six_udenlandsk, six_lov, six_fred,
       six_christ, six_folk],
    name='Sixteenth Century'
)
sev = go.Bar(
    x=['danmark', 'krig', 'peng', 'svensk', 'norsk', 'udenlandsk','lov','fred','christ','folk'],
    y=[sev_danmark, sev_krig, sev_peng, sev_svensk, sev_norsk, sev_udenlandsk, sev_lov, sev_fred,
       sev_christ, sev_folk],
    name='Seventeenth Century'
    )
eig = go.Bar(
    x=['danmark', 'krig', 'peng', 'svensk', 'norsk', 'udenlandsk','lov','fred','christ','folk'],
    y=[eig_danmark, eig_krig, eig_peng, eig_svensk, eig_norsk, eig_udenlandsk, eig_lov, eig_fred,
       eig_christ, eig_folk],
    name='Eighteenth Century'
)
nin = go.Bar(
    x=['danmark', 'krig', 'peng', 'svensk', 'norsk', 'udenlandsk','lov','fred','christ','folk'],
    y=[nin_danmark, nin_krig, nin_peng, nin_svensk, nin_norsk, nin_udenlandsk, nin_lov, nin_fred,
       nin_christ, nin_folk],
    name='Nineteenth Century'
)
data = [six, sev, eig, nin]
layout = go.Layout(
        title='Word Frequency per Century',
        xaxis=dict(
        title='Word'),
        yaxis=dict(
        title='Frequency (% of total words)'),
        barmode='group',
     )
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='bar1')
#============================================================================
#FREQUENCY OF WORDS PER CENTURY
tf6 = defaultdict(int) 
for token in n_6:
        tf6[token] += 1    
tf7 = defaultdict(int) 
for token in n_7:
        tf7[token] += 1
tf8 = defaultdict(int) 
for token in n_8:
        tf8[token] += 1
tf9 = defaultdict(int) 
for token in n_9:
        tf9[token] += 1
#=============================================================================
#STOPWORDS
os.chdir(wd)
stopword = tm.read_txt('stopword_da.txt')
#TOP 50 MOST FREQUENT WORDS OF THE CENTURIES
#SIXTEENTH
dic_6 = dict(tf6)
six_sort = sorted(dic_6.items(), key = operator.itemgetter(1), reverse = True)
top_six = six_sort[:50]
#SEVENTEENTH
dic_7 = dict(tf7)
sev_sort = sorted(dic_7.items(), key = operator.itemgetter(1), reverse = True)
top_sev = sev_sort[:50]
#EIGHTEENTH
dic_8 = dict(tf8)
eig_sort = sorted(dic_8.items(), key = operator.itemgetter(1), reverse = True)
top_eig = eig_sort[:50]
#NINETEENTH
dic_9 = dict(tf9)
nin_sort = sorted(dic_9.items(), key = operator.itemgetter(1), reverse = True)
top_nin = nin_sort[:50]
#============================================================================
#~HERE IS SOME REDUNDANT CODE TO GET THE TOP FREQUENCY WORDS PER CENTURY~
#~HAD TO USE STEM LIST FOR STOP WORD LIST TO WORK~
#~NEED A BETTER STOP WORD LIST FOR DANISH~
#MAKE A DICTIONARY
top6_stem = defaultdict(int) 
for token in nn_6:
        top6_stem[token] += 1 
#GET FREQUENCY AND SAVE TOP 50 IN LIST 
dict_6 = dict(top6_stem)
six_sorted = sorted(dict_6.items(), key = operator.itemgetter(1), reverse = True)
six_stem = six_sorted[:50]
#SAVE ONLY TOKEN, REMOVE STOP WORDS
nn_66 = [x[0] for x in six_stem]
nostop6 = []
for token in nn_66:
    if token not in stopword:
        tokens_nostop = [token]
    nostop6.append(tokens_nostop)
#MAKE FLAT LIST
flat6 = []
for i in nostop6:
    for x in i:
        flat6.append(x)
#MAKE SURE THERE ARE NO REPEATING
nostop6 = set(flat6)
#AND REPEAT...
#--------------------------------------
top7_stem = defaultdict(int) 
for token in nn_7:
        top7_stem[token] += 1 
        
dict_7 = dict(top7_stem)
sev_sorted = sorted(dict_7.items(), key = operator.itemgetter(1), reverse = True)
sev_stem = sev_sorted[:50]
        
nn_77 = [x[0] for x in sev_stem]
nostop7 = []
for token in nn_77:
    if token not in stopword:
        tokens_nostop = [token]
    nostop7.append(tokens_nostop)

flat7 = []
for i in nostop7:
    for x in i:
        flat7.append(x)

nostop7 = set(flat7)
#--------------------------------------
top8_stem = defaultdict(int) 
for token in nn_8:
        top8_stem[token] += 1 
        
dict_8 = dict(top8_stem)
eig_sorted = sorted(dict_8.items(), key = operator.itemgetter(1), reverse = True)
eig_stem = eig_sorted[:50]
        
nn_88 = [x[0] for x in eig_stem]
nostop8 = []
for token in nn_88:
    if token not in stopword:
        tokens_nostop = [token]
    nostop8.append(tokens_nostop)

flat8 = []
for i in nostop8:
    for x in i:
        flat8.append(x)

nostop8 = set(flat8)
#--------------------------------------
top9_stem = defaultdict(int) 
for token in nn_9:
        top9_stem[token] += 1 
        
dict_9 = dict(top9_stem)
nin_sorted = sorted(dict_9.items(), key = operator.itemgetter(1), reverse = True)
nin_stem = nin_sorted[:50]
        
nn_99 = [x[0] for x in nin_stem]
nostop9 = []
for token in nn_99:
    if token not in stopword:
        tokens_nostop = [token]
    nostop9.append(tokens_nostop)

flat9 = []
for i in nostop9:
    for x in i:
        flat9.append(x)

nostop9 = set(flat9)
#=========================================================================
#PRUNING BOTTOM 50 AND TOP 200 WORDS FOR EACH CENTURY
new_tf6 = []
index = 0
for tup in six_sorted:
        if tup[1] >= 50 and tup[1] <200 :
               new_tf6.insert(index, tup)
        index = index + 1
new_tf7 = []
index = 0
for tup in sev_sorted:
        if tup[1] >= 50 and tup[1] <200 :
               new_tf7.insert(index, tup)
        index = index + 1
new_tf8 = []
index = 0
for tup in eig_sorted:
        if tup[1] >= 50 and tup[1] <200 :
               new_tf8.insert(index, tup)
        index = index + 1
new_tf9 = []
index = 0
for tup in nin_sorted:
        if tup[1] >= 50 and tup[1] <200 :
               new_tf9.insert(index, tup)
        index = index + 1     
#SAVE TOP 50 OF THE NEW LIST
new_top6= new_tf6[:50]
new_top7= new_tf7[:50]
new_top8= new_tf8[:50]
new_top9= new_tf9[:50]
#=============================================================================
#FREQUENCY OF TOP 50 DIVIDED BY NUMBER OF TOTAL WORDS, AS A PERCENTAGE
num6 = [str(x[1]/float(len(n_6))*100) for x in new_top6]
num7 = [str(x[1]/float(len(n_7))*100) for x in new_top7]
num8 = [str(x[1]/float(len(n_8))*100) for x in new_top8]
num9 = [str(x[1]/float(len(n_9))*100) for x in new_top9]
#LIST OF TOP 50 WORDS BY THEMSELVES
word6 = [x[0] for x in new_top6]
word7 = [x[0] for x in new_top7]
word8 = [x[0] for x in new_top8]
word9 = [x[0] for x in new_top9]
#==============================================================================
#FANCY PLOT OF EACH CENTURY'S TOP 50 WORDS
sixx = go.Bar(
    x=word6,
    y=num6,
    name='Sixteenth Century'
)
sevv = go.Bar(
    x=word7,
    y=num7,
    name='Seventeenth Century'
    )
eigg = go.Bar(
    x=word8,
    y=num8,
    name='Eighteenth Century'
)
ninn = go.Bar(
    x=word9,
    y=num9,
    name='Nineteenth Century'
)
data = [sixx, sevv, eigg, ninn]
layout = go.Layout(
        title='Top 50 Words per Century',
        xaxis=dict(
        title='Words'),
        yaxis=dict(
        title='Frequency'),
        barmode='group',
     )
fig = go.Figure(data=data, layout=layout)
py.plot(fig, filename='bar2')
#=============================================================================
#~LET'S TAKE A LOOK AT WHAT THE NINETEENTH CENTURY IS LIKE~
#~TOO MANY VALUES TO PLOT, GOTTA CLEAN IT A LITTLE BIT~
new_9 = []
index = 0
for tup in nin_sorted:
        if tup[1] >= 100 and tup[1] <2500 :
               new_9.insert(index, tup)
        index = index + 1  
#ONLY NUMBERS AND ONLY WORDS
a = [i[1] for i in new_9]
b = [i[0] for i in new_9]
# DELETE EVERY SECOND ELEMENT (COUNTING FROM THE FIRST) 
del a[::2]  
del b[::2] 
#MAKE A GRAPH
trace = go.Bar(
        x = b,
        y = a,
        )
data = [trace]
layout = go.Layout(
        title='Word Frequency for Nineteenth Century',
        xaxis=dict(
        title='Words'),
        yaxis=dict(
        title='Frequencies'),
     )
py.plot(data, layout = layout)
#============================================================================
print "THAT'S ALL, FOLKS!"
#============================================================================