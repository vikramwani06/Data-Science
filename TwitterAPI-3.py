import numpy as np
import pandas as pd

import twitter
from twitter import Twitter
from twitter import OAuth
from twitter import TwitterHTTPError
from twitter import TwitterStream

ck= 'hq1ikoFrNoXH32vVwH4tdYewd'
cs= 'UpsUIWyTXNtOoKGCuQZw7dQ9LO2lzx5vxnw069g4gg9BGfWl3Z'
at='753638455-fQJkPQDV4aDafISatU7ZjRBjf4UMb2ufYyKSI2bU'   
ats='Cjym1sTksiwAKp7Zeh78ngdqZYUI29r981Dw4eFqMCJ4e'

oauth= OAuth(at,ats,ck,cs)
twit_api=Twitter(auth=oauth)
t_loc= twit_api.trends.available()
t_loc
ts= TwitterStream(auth=oauth)

iterator = ts.statuses.filter(track="Trump",language="en")

b=[]
for t in iterator:
    print(t)
    b.append(t)
    if len(b)==100:
        break
len(b)


import json
from pandas.io.json import json_normalize

df=json_normalize(b)
df.head()

# Textblob
import textblob as tb
from textblob import TextBlob

!python -m textblob.download_corpora

tweettext=df['text']

tx=tweettext[13]
blob=TextBlob(tx)
tx

blob.sentiment

# Plot Sentiments
import matplotlib.pyplot as plot
%matplotlib inline
polarity=[]
subj=[]
for t in tweettext:
    tx=TextBlob(t)
    polarity.append(tx.sentiment.polarity)
    subj.append(tx.sentiment.subjectivity)

    
poltweet= pd.DataFrame({'polarity':polarity,'subjectivity':subj})   
poltweet.plot(title='Polarity and Subjectivity')


list=[]
list= df['text']
wordstring = list[0]


n=1
while n < 100:
    wordstring += list[n]
    n=n+1
    
    
wordstring
wordlist = wordstring.split()
#wordlist

tweettext=df['text']
#tweettext


blob=TextBlob(wordstring)
blob

blob.sentiment

# Plots
import matplotlib.pyplot as plot
%matplotlib inline
polarity=[]
subj=[]
for t in tweettext:
    tx=TextBlob(t)
    polarity.append(tx.sentiment.polarity)
    subj.append(tx.sentiment.subjectivity)
    
poltweet= pd.DataFrame({'polarity':polarity,'subjectivity':subj})   
poltweet.plot(title='Polarity and Subjectivity')


