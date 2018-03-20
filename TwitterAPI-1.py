import numpy as np
import pandas as pd
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
iterator=ts.statuses.filter(track="Trump", language="en")

b=[]
for t in iterator:
    b.append(t)
    if len(b)==1000:
        break
    len(b)
    
import json
from pandas.io.json import json_normalize

df= json_normalize(b)
df.head()
df.columns


df['entities.hashtags'].head()



df['Text']=df['text']

df_string=df['text'].to_string()

def split_line(column):
    words = column.split()
    for word in words:
        print(word)
print (split_line(df_string),"/n")


print(df_string,"/n")


split_it = df_string.split()
split_it

from collections import Counter

#Counter=Counter(split_it) execute the next block when u uncomment this line
Counter = dict(Counter(split_it).most_common(10))

# Plot of the Top words

import numpy as np
import matplotlib.pyplot as plt

labels, values = zip(*Counter.items())

# sort your values in descending order
indSort = np.argsort(values)[::-1]

# rearrange your data
labels = np.array(labels)[indSort]
values = np.array(values)[indSort]

indexes = np.arange(len(labels))

bar_width = 0.10
fig= plt.subplots(figsize=(8,4))
plt.bar(indexes, values)

# add labels
plt.xticks(indexes + bar_width, labels)
plt.title("Top Words")
plt.show(fig)

#---------------------------------------#

# TextBlob
import textblob as tb
from textblob import TextBlob

!python -m textblob.download_corpora

tweettext=df['text']

# displaying a random tweet
tweettext[10] 

tx=tweettext[10]
blob=TextBlob(tx)
tx

blob.tags

blob.sentences[0]
blob.sentences[0].words

blob.tags
blob.sentences[0].words
blob.noun_phrases
noun=[]

for word,tag in blob.tags:
    if tag == 'NNP':
        noun.append(word)
print(noun) 

#Plotting the nouns
import matplotlib.pyplot as plt
% matplotlib inline
wordlist=pd.DataFrame()

for u in noun:
    wsplit=u.split()
    wordlist=wordlist.append(wsplit,ignore_index=True)
wordlist.shape    
wordlist.head()
allword=wordlist.groupby(0).size()
allword.head()
top20nouns=allword.sort_values(0,ascending=False)
top20nouns.plot(kind='bar',title='Top nouns')
                                           
# Plot top nouns from 100 Tweets
list=[]
list= df['text']
wordstring = list[0]

n=1
while n < 100:
    wordstring += list[n]
    n=n+1

wordstring
wordlist = wordstring.split()
wordlist

#tx=list[1]
blob=TextBlob(wordstring)
tx
blob

tweettext= df['text']
tweettext

blob.tags
blob.sentences[0].words
blob.noun_phrases
nouns= []
for word,tag in blob.tags:
     if tag == 'NNP':
            nouns.append(word)   
nouns 

counter = {}
for i in nouns: counter[i] = counter.get(i, 0) + 1
counter
POPULARwords= sorted([ (freq,word) for word, freq in counter.items() ], reverse=True)[:30]
    
#POPULARwords


import pandas as pd

labels=['Count','Words']
df1= pd.DataFrame.from_records(POPULARwords, columns=labels)
df1.head()

import seaborn as sns
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(15, 7)) # defining the size of figure
sns.barplot(x='Words', y='Count', data=df1,palette='muted',ax=ax) # creating barplot of 'Runs scored or allowed' versus Date of each relevant game
#sns.despine() # Removing the top and right spines from plot
plt.show(fig)

# ----------------End of Lab2-------



list(df.columns)
df.shape

df['text']
df['id']
df['created_at']
df['user.id']
df['user.name']
df['user.screen_name']
df['entities.hashtags']

twit_api=Twitter(auth=oauth)
t_loc = twit_api.trends.available()
t_loc_df=json_normalize(t_loc)

t_loc_df.head()
t_loc_df[t_loc_df['country']=='India']


la_trends=twit_api.trends.place(_id= 23424848)
la_trends
la_df=json_normalize(la_trends,'trends')
la_df.head()

res= twit_api.search.tweets(q='Bitcoin', count=100)
resdf= json_normalize(res, 'statuses')
list(resdf.columns)

resdf['text']

res2=twit_api.search.tewwts(q="Bitcoin", count=100, until="2018-02-05")
resdf2=json_normalize(res2, 'statuses')

resdf['created_at'].min()
resdf2['created_at'].min()