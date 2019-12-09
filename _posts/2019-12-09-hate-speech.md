---
title: "Hate Speech Detection"
date: 2019-12-09
tags: [hate speech, classification, machine learning, deep learning, data science]
header:
  image: "/images/perceptron/percept.jpg"
excerpt: "Hate Speech, Classification, Machine Learning, Deep Learning"
mathjax: "true"
---


# Abstract

In this era of the digital age, online hate speech residing in social media networks can influence hate violence or even crimes towards a certain group of people. Hate related attacks targetted at specific groups of people are at a 16-year high in the United States of America, statistics released by the FBI reported. [[1]](https://www.nytimes.com/2019/11/12/us/hate-crimes-fbi-report.html) Therefore, there is a growing need to eradicate hate speech as much as possible through automatic detection to ease the load on moderators.

Datasets were obtained from Reddit and a white supremacist forum, Gab where there contains human labelled comments that are determined as hate speech related. [[2]](https://github.com/jing-qian/A-Benchmark-Dataset-for-Learning-to-Intervene-in-Online-Hate-Speech)

Multiple modelling approaches will be explored, such as machine learning models and even state-of-the-art deep learning models. F1 score and recall will be the metrics to be prioritised in model comparison. In the event where both are the same, actual False Negatives and False Postive numbers will be looked at.

# Problem Statement

In this digital age, online hate speech has increased over the past few years. Studies has shown that online hate speech can lead to offline violence towards a certain group. [[3]](https://phys.org/news/2019-10-online-speech-crimes-minorities.html)

In some cases, social media can lead to a more direct role, in this case the New Zealand shooting incident was broadcasted live on Facebook.[[4]](https://www.nytimes.com/2019/03/14/world/asia/new-zealand-shooting-updates-christchurch.html)

Due to the societal concern and how widespread hate speech is becoming on the Internet  and especially on social media, there is a strong need to classify online hate speech comments that are considered hate speech. [[5]](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0221152#sec001)

Hate speech definition: Hate speech is speech that attacks a person or a group on the basis of protected attributes such as race, religion, ethnic origin, national origin, sex, disability, sexual orientation, or gender identity. [[6]](https://en.wikipedia.org/wiki/Hate_speech)

Hate speech categories:
- misogyny --> aimed at women
- misandry --> aimed at men
- racism --> aimed at specific race
- sexual orientation
- religion
- disability

## EDA

### Word Cloud


```python
#specifying own stopwords
stopwords = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren", "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't", "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i", "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn", "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t", "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they", "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we", "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won", "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought", "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're", "we've", "what's", "when's", "where's", "who's", "why's", "would"] \
+ ['was', 'really', 'let', 'like', 'also', 'dankMemes', 'imgoingtohellforthis', 'KotakuInAction', 'MensRights', 'MetaCanada', 'MGTOW'\
  'PussyPass', 'PussyPassDenied', 'The_Donald', 'TumblrInAction', 'please', 'moderators', 'questions', 'concerns', 'contact', 'action'\
  'perform', 'bot', 'subreddit', 'dankmemes', 'kotakuinaction', 'mensrights', 'metacanada', 'mgtowpussypass', 'pussypassdenied', \
   'the_donald', 'tumblrinaction', 'pussy', 'pass']
stopwords = set(stopwords)
```


```python
wordcloud = WordCloud(stopwords=stopwords, background_color="white", max_font_size=80, max_words=20)

#text has to be one single string
all_text =' '.join([txt for txt in df.loc[:,'text']]).lower()
wordcloud.generate(all_text)
plt.figure(figsize=(10,8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()
```


![png](Hate-speech-detection-blogpost_files/Hate-speech-detection-blogpost_8_0.png)


### Top Unigrams


```python
def get_top_n_unigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
```


```python
common_words = get_top_n_unigram(df.loc[:,'text'], 20)

unigram_df = pd.DataFrame(common_words, columns = ['text' , 'count'])

plt.figure(figsize=(12, 9));
unigram_df.groupby('text').sum()['count'].sort_values(ascending=True).plot(
    kind='barh');
plt.ylabel('');
plt.title('Top 20 unigrams', fontdict={'fontsize': 30});
#set large enough font size for ytick labels
plt.gca().tick_params(axis='y', labelsize=16);
```


![png](Hate-speech-detection-blogpost_files/Hate-speech-detection-blogpost_11_0.png)


### Top Bigrams


```python
def get_top_n_bigram(corpus, n=None):
    vec = CountVectorizer(ngram_range=(2, 2), stop_words=stopwords).fit(corpus)
    bag_of_words = vec.transform(corpus)
    sum_words = bag_of_words.sum(axis=0)
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    return words_freq[:n]
```


```python
common_words = get_top_n_bigram(df.loc[:,'text'], 20)

bigram_df = pd.DataFrame(common_words, columns = ['text' , 'count'])

plt.figure(figsize=(12, 9));
bigram_df.groupby('text').sum()['count'].sort_values(ascending=True).plot(
    kind='barh');
plt.ylabel('');
plt.title('Top 20 bigrams', fontdict={'fontsize': 30});
#set large enough font size for ytick labels
plt.gca().tick_params(axis='y', labelsize=16);
```


![png](Hate-speech-detection-blogpost_files/Hate-speech-detection-blogpost_14_0.png)


# Modelling

## BOW Modelling

### Pipeline


```python
from nltk.corpus import stopwords
stopwords_nltk =  set(stopwords.words('english'))

def superPipeline(Dataframes,Vectorizerlist,ClassifierList,Dfnames,pipe_params,methodgridname, df_column):
    Methodgrid=[]
    metnum=len(Dataframes)*len(Vectorizerlist)*len(ClassifierList)
    n=0
    for index,df in enumerate(Dataframes):
        X=Dataframes[index][df_column]
        y=Dataframes[index]['hate']
        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=28)

        for Vectorizer in Vectorizerlist:
            for Classifier in ClassifierList:
                n+=1
                print(f'{n} of {metnum} of methods attempting')
                method={}
                pipe = Pipeline([
                    ('vec', Vectorizer ),
                    ('class', Classifier)
                ])

                gs = GridSearchCV(pipe, param_grid=pipe_params, cv=5,verbose=1,n_jobs=-1, scoring='f1')
                gs.fit(X_train, y_train)
                method=(gs.best_params_)
                method['Cross_Val_Score']=(gs.best_score_)
                method['Test_Score']=gs.score(X_test,y_test)
                method['Vectorizer']=str(Vectorizer).split('(')[0]
                method['Data']=str(Dfnames[index])
                method['Classifier']=str(Classifier).split('(')[0]
                Methodgrid.append(method)

                tn, fp, fn, tp = confusion_matrix(y_test, gs.predict(X_test)).ravel()
                print(f"{str(Classifier).split('(')[0]} Confusion Matrix:")
                print(f"True Negatives: {tn}")
                print(f"False Positives: {fp}")
                print(f"False Negatives: {fn}")
                print(f"True Positives: {tp}")
                print('\n')

                report = classification_report(y_test, gs.predict(X_test), target_names=['Predict 0', 'Predict 1'], output_dict=True)
                class_table = pd.DataFrame(report).transpose()
                display(class_table)

    Methodgrid=pd.DataFrame(Methodgrid)
    Methodgrid.to_csv(methodgridname,index=False)
    return Methodgrid
```

#### Choosing best vectorizer


```python
dataframes=[df]
df_names = ['df']
vectorizer_lst = [TfidfVectorizer(),CountVectorizer()]
classifier_lst = [LogisticRegression(), MultinomialNB()]
pipe_params = {
                    'vec__max_features': [int(i) for i in np.linspace(5000,20000,4)],
                    'vec__min_df': [2],
                    'vec__max_df': [.95],
                    'vec__ngram_range': [(1,1),(1,2)],
                    'vec__stop_words':[stopwords_nltk]
                }
superPipeline(dataframes, vectorizer_lst, classifier_lst, df_names, pipe_params, 'vectorizer_grid.csv')
```

    1 of 4 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  1.1min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    LogisticRegression Confusion Matrix:
    True Negatives: 7364
    False Positives: 322
    False Negatives: 1136
    True Positives: 3769


    2 of 4 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  1.0min finished


    MultinomialNB Confusion Matrix:
    True Negatives: 7188
    False Positives: 498
    False Negatives: 1804
    True Positives: 3101


    3 of 4 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  1.4min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    LogisticRegression Confusion Matrix:
    True Negatives: 7219
    False Positives: 467
    False Negatives: 925
    True Positives: 3980


    4 of 4 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  1.3min finished


    MultinomialNB Confusion Matrix:
    True Negatives: 6309
    False Positives: 1377
    False Negatives: 874
    True Positives: 4031







<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier</th>
      <th>Cross_Val_Score</th>
      <th>Data</th>
      <th>Test_Score</th>
      <th>Vectorizer</th>
      <th>vec__max_df</th>
      <th>vec__max_features</th>
      <th>vec__min_df</th>
      <th>vec__ngram_range</th>
      <th>vec__stop_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>0.882397</td>
      <td>df</td>
      <td>0.884203</td>
      <td>TfidfVectorizer</td>
      <td>0.95</td>
      <td>5000</td>
      <td>2</td>
      <td>(1, 1)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MultinomialNB</td>
      <td>0.810966</td>
      <td>df</td>
      <td>0.817171</td>
      <td>TfidfVectorizer</td>
      <td>0.95</td>
      <td>5000</td>
      <td>2</td>
      <td>(1, 2)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>LogisticRegression</td>
      <td>0.886686</td>
      <td>df</td>
      <td>0.889445</td>
      <td>CountVectorizer</td>
      <td>0.95</td>
      <td>20000</td>
      <td>2</td>
      <td>(1, 1)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MultinomialNB</td>
      <td>0.821159</td>
      <td>df</td>
      <td>0.821222</td>
      <td>CountVectorizer</td>
      <td>0.95</td>
      <td>5000</td>
      <td>2</td>
      <td>(1, 1)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
  </tbody>
</table>
</div>



Generally CountVectorizer is better and it does better on unigrams. This is probably because of the type of words that are being used to classify if it is hate speech or not.

#### Choosing best vectorizer with SVC


```python
dataframes=[df]
df_names = ['df']
vectorizer_lst = [TfidfVectorizer(),CountVectorizer()]
classifier_lst = [SVC()]
pipe_params = {
                    'vec__max_features': [int(i) for i in np.linspace(5000,20000,4)],
                    'vec__min_df': [2],
                    'vec__max_df': [.95],
                    'vec__ngram_range': [(1,1),(1,2),(1,3)],
                    'vec__stop_words':[stopwords_nltk]
                }
superPipeline(dataframes, vectorizer_lst, classifier_lst, df_names, pipe_params, 'vectorizer_grid_linearmodels.csv', 'tok_lemma')
```

    1 of 2 of methods attempting
    Fitting 5 folds for each of 12 candidates, totalling 60 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    /Users/clementow/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      "timeout or by a memory leak.", UserWarning
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed: 39.0min
    [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed: 54.3min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)


    SVC Confusion Matrix:
    True Negatives: 7686
    False Positives: 0
    False Negatives: 4905
    True Positives: 0




    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples.
      'precision', 'predicted', average, warn_for)



<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.758100</td>
      <td>0.610436</td>
      <td>1.000000</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>4905.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.610436</td>
      <td>0.610436</td>
      <td>0.610436</td>
      <td>0.610436</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.379050</td>
      <td>0.305218</td>
      <td>0.500000</td>
      <td>12591.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.462772</td>
      <td>0.372632</td>
      <td>0.610436</td>
      <td>12591.000000</td>
    </tr>
  </tbody>
</table>
</div>


    2 of 2 of methods attempting
    Fitting 5 folds for each of 12 candidates, totalling 60 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    /Users/clementow/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      "timeout or by a memory leak.", UserWarning
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed: 44.8min
    [Parallel(n_jobs=-1)]: Done  60 out of  60 | elapsed: 64.7min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    SVC Confusion Matrix:
    True Negatives: 7624
    False Positives: 62
    False Negatives: 3498
    True Positives: 1407





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.810719</td>
      <td>0.685488</td>
      <td>0.991933</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.441481</td>
      <td>0.957794</td>
      <td>0.286850</td>
      <td>4905.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.717258</td>
      <td>0.717258</td>
      <td>0.717258</td>
      <td>0.717258</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.626100</td>
      <td>0.821641</td>
      <td>0.639392</td>
      <td>12591.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.666877</td>
      <td>0.791569</td>
      <td>0.717258</td>
      <td>12591.000000</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier</th>
      <th>Cross_Val_Score</th>
      <th>Data</th>
      <th>Test_Score</th>
      <th>Vectorizer</th>
      <th>vec__max_df</th>
      <th>vec__max_features</th>
      <th>vec__min_df</th>
      <th>vec__ngram_range</th>
      <th>vec__stop_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SVC</td>
      <td>0.000000</td>
      <td>df</td>
      <td>0.000000</td>
      <td>TfidfVectorizer</td>
      <td>0.95</td>
      <td>5000</td>
      <td>2</td>
      <td>(1, 1)</td>
      <td>{doesn't, under, was, were, down, against, out...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SVC</td>
      <td>0.258985</td>
      <td>df</td>
      <td>0.441481</td>
      <td>CountVectorizer</td>
      <td>0.95</td>
      <td>5000</td>
      <td>2</td>
      <td>(1, 3)</td>
      <td>{doesn't, under, was, were, down, against, out...</td>
    </tr>
  </tbody>
</table>
</div>



With SVM Classifier (SVC), it is the same as the other classifiers above where CountVectorizer works better.

#### Choosing best model based on CountVectorizer

With CountVectorizer as the determined vectorizer, it is time to choose the best model that works well with it.


```python
def superPipeline(Dataframes,Vectorizerlist,ClassifierList,Dfnames,pipe_params,methodgridname, df_column):
'''
Function that handles the pipeline to match each vectorizer with each classifer with their corresponding
parameters for GridSearch.
'''
    Methodgrid=[]
    metnum=len(Dataframes)*len(Vectorizerlist)*len(ClassifierList)
    n=0
    for index,df in enumerate(Dataframes):
        X=Dataframes[index][df_column]
        y=Dataframes[index]['hate']
        X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=28)

        for Vectorizer in Vectorizerlist:
            for Classifier in ClassifierList:
                n+=1
                print(f'{n} of {metnum} of methods attempting')
                method={}
                pipe = Pipeline([
                    ('vec', Vectorizer ),
                    ('class', Classifier)
                ])

                gs = GridSearchCV(pipe, param_grid=pipe_params, cv=5,verbose=1,n_jobs=1, scoring='f1')
                gs.fit(X_train, y_train)
                method=(gs.best_params_)
                method['Cross_Val_Score']=(gs.best_score_)
                method['Test_Score']=gs.score(X_test,y_test)
                method['Vectorizer']=str(Vectorizer).split('(')[0]
                method['Data']=str(Dfnames[index])
                method['Classifier']=str(Classifier).split('(')[0]
                Methodgrid.append(method)

                tn, fp, fn, tp = confusion_matrix(y_test, gs.predict(X_test)).ravel()
                print(f"{str(Classifier).split('(')[0]} Confusion Matrix:")
                print(f"True Negatives: {tn}")
                print(f"False Positives: {fp}")
                print(f"False Negatives: {fn}")
                print(f"True Positives: {tp}")
                print('\n')

                report = classification_report(y_test, gs.predict(X_test), target_names=['Predict 0', 'Predict 1'], output_dict=True)
                class_table = pd.DataFrame(report).transpose()
                display(class_table)

    Methodgrid=pd.DataFrame(Methodgrid)
    Methodgrid.to_csv(methodgridname,index=False)
    return Methodgrid
```


```python
dataframes=[df]
df_names = ['df']
vectorizer_lst = [CountVectorizer()]
classifier_lst = [LogisticRegression(), MultinomialNB(), ExtraTreesClassifier()]
pipe_params = {
                    'vec__max_features': [int(i) for i in np.linspace(5000,20000,4)],
                    'vec__min_df': [2],
                    'vec__max_df': [.95],
                    'vec__ngram_range': [(1,1),(1,2)],
                    'vec__stop_words':[stopwords_nltk]
                }
superPipeline(dataframes, vectorizer_lst, classifier_lst, df_names, pipe_params, 'class_grid.csv', 'lemma')
```

    1 of 3 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  1.6min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    LogisticRegression Confusion Matrix:
    True Negatives: 7219
    False Positives: 467
    False Negatives: 925
    True Positives: 3980





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.912066</td>
      <td>0.886419</td>
      <td>0.939240</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.851155</td>
      <td>0.894985</td>
      <td>0.811417</td>
      <td>4905.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.889445</td>
      <td>0.889445</td>
      <td>0.889445</td>
      <td>0.889445</td>
    </tr>
  </tbody>
</table>
</div>


    2 of 3 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:   59.2s finished


    MultinomialNB Confusion Matrix:
    True Negatives: 6309
    False Positives: 1377
    False Negatives: 874
    True Positives: 4031





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.848611</td>
      <td>0.878324</td>
      <td>0.820843</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.781732</td>
      <td>0.745377</td>
      <td>0.821814</td>
      <td>4905.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.821222</td>
      <td>0.821222</td>
      <td>0.821222</td>
      <td>0.821222</td>
    </tr>
  </tbody>
</table>
</div>


    3 of 3 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  3.9min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/ensemble/forest.py:245: FutureWarning: The default value of n_estimators will change from 10 in version 0.20 to 100 in 0.22.
      "10 in version 0.20 to 100 in 0.22.", FutureWarning)


    ExtraTreesClassifier Confusion Matrix:
    True Negatives: 7092
    False Positives: 594
    False Negatives: 1048
    True Positives: 3857





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.896247</td>
      <td>0.871253</td>
      <td>0.922717</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.824498</td>
      <td>0.866547</td>
      <td>0.786340</td>
      <td>4905.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.869589</td>
      <td>0.869589</td>
      <td>0.869589</td>
      <td>0.869589</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier</th>
      <th>Cross_Val_Score</th>
      <th>Data</th>
      <th>Test_Score</th>
      <th>Vectorizer</th>
      <th>vec__max_df</th>
      <th>vec__max_features</th>
      <th>vec__min_df</th>
      <th>vec__ngram_range</th>
      <th>vec__stop_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>0.886686</td>
      <td>df</td>
      <td>0.889445</td>
      <td>CountVectorizer</td>
      <td>0.95</td>
      <td>20000</td>
      <td>2</td>
      <td>(1, 1)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MultinomialNB</td>
      <td>0.821159</td>
      <td>df</td>
      <td>0.821222</td>
      <td>CountVectorizer</td>
      <td>0.95</td>
      <td>5000</td>
      <td>2</td>
      <td>(1, 1)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>ExtraTreesClassifier</td>
      <td>0.866697</td>
      <td>df</td>
      <td>0.869589</td>
      <td>CountVectorizer</td>
      <td>0.95</td>
      <td>15000</td>
      <td>2</td>
      <td>(1, 2)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
  </tbody>
</table>
</div>



As we can see the best F1 score still goes to LogisticRegression at `88.16%` and unigram CountVectorizer.
The model with the lowest False Negatives is Multinomial Naive-Bayes whoever the score is the worse out of the above classifiers.
Coming in second for the lowest False Negatives goes to the LogisticRegression model.

#### SVC Optimization

So far LogisticRegression model works the best for this dataset classifying whether a comment is hate speech or not.
Let's use another linear based model to see if the score can further improved.


```python
dataframes=[df]
df_names = ['df']
vectorizer_lst = [CountVectorizer(max_df=0.95, min_df=2, ngram_range=(1,1))]
classifier_lst = [SVC()]
pipe_params = {
                'class__C':[0.1,1,10]
                }

superPipeline(dataframes, vectorizer_lst, classifier_lst, df_names, pipe_params, 'svm.csv', 'tok_lemma')
```

    1 of 1 of methods attempting
    Fitting 5 folds for each of 3 candidates, totalling 15 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/metrics/classification.py:1437: UndefinedMetricWarning: F-score is ill-defined and being set to 0.0 due to no predicted samples.
      'precision', 'predicted', average, warn_for)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)
    [Parallel(n_jobs=1)]: Done  15 out of  15 | elapsed: 53.8min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:193: FutureWarning: The default value of gamma will change from 'auto' to 'scale' in version 0.22 to account better for unscaled features. Set gamma explicitly to 'auto' or 'scale' to avoid this warning.
      "avoid this warning.", FutureWarning)


    SVC Confusion Matrix:
    True Negatives: 7407
    False Positives: 279
    False Negatives: 1291
    True Positives: 3614





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.904175</td>
      <td>0.851575</td>
      <td>0.963700</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.821550</td>
      <td>0.928333</td>
      <td>0.736799</td>
      <td>4905.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.875308</td>
      <td>0.875308</td>
      <td>0.875308</td>
      <td>0.875308</td>
    </tr>
    <tr>
      <th>macro avg</th>
      <td>0.862863</td>
      <td>0.889954</td>
      <td>0.850250</td>
      <td>12591.000000</td>
    </tr>
    <tr>
      <th>weighted avg</th>
      <td>0.871987</td>
      <td>0.881477</td>
      <td>0.875308</td>
      <td>12591.000000</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier</th>
      <th>Cross_Val_Score</th>
      <th>Data</th>
      <th>Test_Score</th>
      <th>Vectorizer</th>
      <th>class__C</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>SVC</td>
      <td>0.792316</td>
      <td>df</td>
      <td>0.82155</td>
      <td>CountVectorizer</td>
      <td>10</td>
    </tr>
  </tbody>
</table>
</div>



The SVC model has an F1 score of `86.29%` which is considered a decent score but still short of the LogisticRegression performance (`88.16%` before optimization). Though it did better in terms of False Positives, it failed to detect many comments which were considered hate speech which resulted in a high occurence of False Negatives.

#### Ensemble models

Next we will try ensemble models to see how well it does and if it can be a decent contender to LogisticRegression so far.


```python
dataframes=[df]
df_names = ['df']
vectorizer_lst = [CountVectorizer()]
classifier_lst = [RandomForestClassifier(n_estimators=100), GradientBoostingClassifier(n_estimators=100), AdaBoostClassifier(n_estimators=100)]
pipe_params = {
                    'vec__max_features': [int(i) for i in np.linspace(5000,20000,4)],
                    'vec__min_df': [2],
                    'vec__max_df': [.95],
                    'vec__ngram_range': [(1,1),(1,2)],
                    'vec__stop_words':[stopwords_nltk]
                }
superPipeline(dataframes, vectorizer_lst, classifier_lst, df_names, pipe_params, 'class_ensemble_grid.csv')
```

    1 of 3 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    /usr/local/lib/python3.6/dist-packages/joblib/externals/loky/process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      "timeout or by a memory leak.", UserWarning
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed: 34.0min finished


    RandomForestClassifier Confusion Matrix:
    True Negatives: 7126
    False Positives: 559
    False Negatives: 842
    True Positives: 4064





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.894327</td>
      <td>0.927261</td>
      <td>0.910496</td>
      <td>7685.00000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.879083</td>
      <td>0.828373</td>
      <td>0.852975</td>
      <td>4906.00000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.888730</td>
      <td>0.888730</td>
      <td>0.888730</td>
      <td>0.88873</td>
    </tr>
  </tbody>
</table>
</div>


    2 of 3 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    /usr/local/lib/python3.6/dist-packages/joblib/externals/loky/process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      "timeout or by a memory leak.", UserWarning
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  5.9min finished


    GradientBoostingClassifier Confusion Matrix:
    True Negatives: 7329
    False Positives: 356
    False Negatives: 1016
    True Positives: 3890





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.878250</td>
      <td>0.953676</td>
      <td>0.914410</td>
      <td>7685.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.916156</td>
      <td>0.792907</td>
      <td>0.850087</td>
      <td>4906.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.891033</td>
      <td>0.891033</td>
      <td>0.891033</td>
      <td>0.891033</td>
    </tr>
  </tbody>
</table>
</div>


    3 of 3 of methods attempting
    Fitting 5 folds for each of 8 candidates, totalling 40 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 2 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  40 out of  40 | elapsed:  4.0min finished


    AdaBoostClassifier Confusion Matrix:
    True Negatives: 7282
    False Positives: 403
    False Negatives: 927
    True Positives: 3979





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.887075</td>
      <td>0.947560</td>
      <td>0.916321</td>
      <td>7685.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.908033</td>
      <td>0.811048</td>
      <td>0.856804</td>
      <td>4906.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.894369</td>
      <td>0.894369</td>
      <td>0.894369</td>
      <td>0.894369</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>vec__max_df</th>
      <th>vec__max_features</th>
      <th>vec__min_df</th>
      <th>vec__ngram_range</th>
      <th>vec__stop_words</th>
      <th>Cross_Val_Score</th>
      <th>Test_Score</th>
      <th>Vectorizer</th>
      <th>Data</th>
      <th>Classifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.95</td>
      <td>15000</td>
      <td>2</td>
      <td>(1, 2)</td>
      <td>{hers, you'll, ain, being, shouldn, isn't, aga...</td>
      <td>0.861168</td>
      <td>0.852975</td>
      <td>CountVectorizer</td>
      <td>df</td>
      <td>RandomForestClassifier</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.95</td>
      <td>20000</td>
      <td>2</td>
      <td>(1, 1)</td>
      <td>{hers, you'll, ain, being, shouldn, isn't, aga...</td>
      <td>0.856908</td>
      <td>0.850087</td>
      <td>CountVectorizer</td>
      <td>df</td>
      <td>GradientBoostingClassifier</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.95</td>
      <td>10000</td>
      <td>2</td>
      <td>(1, 2)</td>
      <td>{hers, you'll, ain, being, shouldn, isn't, aga...</td>
      <td>0.861670</td>
      <td>0.856804</td>
      <td>CountVectorizer</td>
      <td>df</td>
      <td>AdaBoostClassifier</td>
    </tr>
  </tbody>
</table>
</div>



Adaboost does the best among all the ensemble models with a F1 score of `88.66%`. However we want to ensure the model does well to reduce false negatives, hence RandomForest classifier is superior with a lower number of false negatives and a decent F1 score of `88.17%`. This slightly edges LogisticRegression (before optimization) by `0.01%`.

#### RandomForest optimization

Since RandomForest and LogisticRegression are the top 2 models edging very close with each other. We shall do some optimization of parameters for each model.


```python
dataframes=[df]
df_names = ['df']
vectorizer_lst = [CountVectorizer(max_features=15000, max_df=0.95, min_df=2, ngram_range=(1,2), stop_words=stopwords_nltk)]
classifier_lst = [RandomForestClassifier()]
pipe_params = {
               'class__n_estimators': [10, 100, 200],
               'class__max_depth': [None, 1, 3, 5, 7, 9],
               'class__max_features': [3, 5, 6]
                }
superPipeline(dataframes, vectorizer_lst, classifier_lst, df_names, pipe_params, 'class_rf_grid.csv')
```

    1 of 1 of methods attempting
    Fitting 5 folds for each of 36 candidates, totalling 180 fits


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.
    [Parallel(n_jobs=1)]: Done 180 out of 180 | elapsed: 76.4min finished


    RandomForestClassifier Confusion Matrix:
    True Negatives: 7448
    False Positives: 237
    False Negatives: 1877
    True Positives: 3029





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.798713</td>
      <td>0.969161</td>
      <td>0.875720</td>
      <td>7685.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.927434</td>
      <td>0.617407</td>
      <td>0.741312</td>
      <td>4906.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.832102</td>
      <td>0.832102</td>
      <td>0.832102</td>
      <td>0.832102</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>class__max_depth</th>
      <th>class__max_features</th>
      <th>class__n_estimators</th>
      <th>Cross_Val_Score</th>
      <th>Test_Score</th>
      <th>Vectorizer</th>
      <th>Data</th>
      <th>Classifier</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>None</td>
      <td>6</td>
      <td>200</td>
      <td>0.735024</td>
      <td>0.741312</td>
      <td>CountVectorizer</td>
      <td>df</td>
      <td>RandomForestClassifier</td>
    </tr>
  </tbody>
</table>
</div>



With an increased number of estimators, we want to ensure that it generalises well with unseen data. However, it compromises on the F1 score and recall, esp on classifying hate speech.

#### Logistic Regression optimization


```python
dataframes=[df]
df_names = ['df']
vectorizer_lst = [CountVectorizer()]
classifier_lst = [LogisticRegression()]
pipe_params = {

                    'vec__max_features': [10000,15000,17500,None],
                    'vec__min_df': [2,3],
                    'vec__max_df': [.95,.9],
                    'vec__ngram_range': [(1,1), (1,2),(1,3)],
                    'vec__stop_words':[stopwords_nltk],
                    'class__C':[0.1,1,10]
                }

superPipeline(dataframes, vectorizer_lst, classifier_lst, df_names, pipe_params, 'logreg.csv')
```

    1 of 1 of methods attempting
    Fitting 5 folds for each of 144 candidates, totalling 720 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  2.1min
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  8.8min
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed: 22.8min
    [Parallel(n_jobs=-1)]: Done 720 out of 720 | elapsed: 44.0min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    LogisticRegression Confusion Matrix:
    True Negatives: 7319
    False Positives: 367
    False Negatives: 949
    True Positives: 3956





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.917513</td>
      <td>0.885220</td>
      <td>0.952251</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.857391</td>
      <td>0.915105</td>
      <td>0.806524</td>
      <td>4905.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.895481</td>
      <td>0.895481</td>
      <td>0.895481</td>
      <td>0.895481</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier</th>
      <th>Cross_Val_Score</th>
      <th>Data</th>
      <th>Test_Score</th>
      <th>Vectorizer</th>
      <th>class__C</th>
      <th>vec__max_df</th>
      <th>vec__max_features</th>
      <th>vec__min_df</th>
      <th>vec__ngram_range</th>
      <th>vec__stop_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>0.853168</td>
      <td>df</td>
      <td>0.857391</td>
      <td>CountVectorizer</td>
      <td>0.1</td>
      <td>0.95</td>
      <td>10000</td>
      <td>3</td>
      <td>(1, 1)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
  </tbody>
</table>
</div>



After optimization, LogisticRegression is the superior model with a better F1 score with its results being more interpretable. It also generalises better with unseen data as compared to RandomForest.

#### Modelling using POS Tags

By classifying hate speech on our dataset, we want to see if there is any relation to how the insults are structured grammatically, which can be used as our features for classification.

##### Pre-processing


```python
X = df['pos']
y = df['hate']
```


```python
X_train_pos, X_test_pos, y_train_pos, y_test_pos = train_test_split(X, y, stratify=y, random_state=28)
```


```python
y_train_pos.value_counts(normalize=True)
```




    0    0.610389
    1    0.389611
    Name: hate, dtype: float64



##### Modelling

We want to see if there are any relations to effective clasification by increasing the ngram range for the vectorizers.


```python
dataframes=[df]
df_names = ['df']
vectorizer_lst = [TfidfVectorizer(), CountVectorizer()]
classifier_lst = [LogisticRegression()]
pipe_params = {

                    'vec__max_features': [10000,15000,17500,None],
                    'vec__min_df': [2,3],
                    'vec__max_df': [.95,.9],
                    'vec__ngram_range': [(1,1), (1,2),(1,3), (1,4), (1,5)],
                    'vec__stop_words':[stopwords_nltk],
                    'class__C':[0.1,1,10]
                }

superPipeline(dataframes, vectorizer_lst, classifier_lst, df_names, pipe_params, 'logreg_pos_ngram.csv', 'pos')
```

    1 of 2 of methods attempting
    Fitting 5 folds for each of 240 candidates, totalling 1200 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  2.1min
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed:  8.7min
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed: 20.2min
    [Parallel(n_jobs=-1)]: Done 792 tasks      | elapsed: 38.1min
    /Users/clementow/anaconda3/lib/python3.7/site-packages/joblib/externals/loky/process_executor.py:706: UserWarning: A worker stopped while some jobs were given to the executor. This can be caused by a too short worker timeout or by a memory leak.
      "timeout or by a memory leak.", UserWarning
    [Parallel(n_jobs=-1)]: Done 1200 out of 1200 | elapsed: 63.3min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)


    LogisticRegression Confusion Matrix:
    True Negatives: 5701
    False Positives: 1985
    False Negatives: 2935
    True Positives: 1970





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.698566</td>
      <td>0.660144</td>
      <td>0.741738</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.444695</td>
      <td>0.498104</td>
      <td>0.401631</td>
      <td>4905.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.609245</td>
      <td>0.609245</td>
      <td>0.609245</td>
      <td>0.609245</td>
    </tr>
  </tbody>
</table>
</div>


    2 of 2 of methods attempting
    Fitting 5 folds for each of 240 candidates, totalling 1200 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  4.0min
    [Parallel(n_jobs=-1)]: Done 192 tasks      | elapsed: 20.3min
    [Parallel(n_jobs=-1)]: Done 442 tasks      | elapsed: 52.8min
    [Parallel(n_jobs=-1)]: Done 792 tasks      | elapsed: 90.7min
    [Parallel(n_jobs=-1)]: Done 1200 out of 1200 | elapsed: 145.6min finished
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)
    /Users/clementow/anaconda3/lib/python3.7/site-packages/sklearn/svm/base.py:929: ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.
      "the number of iterations.", ConvergenceWarning)


    LogisticRegression Confusion Matrix:
    True Negatives: 6244
    False Positives: 1442
    False Negatives: 3388
    True Positives: 1517





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>f1-score</th>
      <th>precision</th>
      <th>recall</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Predict 0</th>
      <td>0.721099</td>
      <td>0.648256</td>
      <td>0.812386</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <th>Predict 1</th>
      <td>0.385809</td>
      <td>0.512673</td>
      <td>0.309276</td>
      <td>4905.000000</td>
    </tr>
    <tr>
      <th>accuracy</th>
      <td>0.616393</td>
      <td>0.616393</td>
      <td>0.616393</td>
      <td>0.616393</td>
    </tr>
  </tbody>
</table>
</div>





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Classifier</th>
      <th>Cross_Val_Score</th>
      <th>Data</th>
      <th>Test_Score</th>
      <th>Vectorizer</th>
      <th>class__C</th>
      <th>vec__max_df</th>
      <th>vec__max_features</th>
      <th>vec__min_df</th>
      <th>vec__ngram_range</th>
      <th>vec__stop_words</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>LogisticRegression</td>
      <td>0.447407</td>
      <td>df</td>
      <td>0.444695</td>
      <td>TfidfVectorizer</td>
      <td>10</td>
      <td>0.95</td>
      <td>10000</td>
      <td>3</td>
      <td>(1, 5)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>LogisticRegression</td>
      <td>0.413341</td>
      <td>df</td>
      <td>0.385809</td>
      <td>CountVectorizer</td>
      <td>1</td>
      <td>0.95</td>
      <td>10000</td>
      <td>3</td>
      <td>(1, 3)</td>
      <td>{whom, me, until, m, couldn, you'd, her, but, ...</td>
    </tr>
  </tbody>
</table>
</div>



The above gridsearch results shows that there is no obvious relation that the model can learn from in terms of POS tags of the sentences in the hate speech comments. This is likely due to the fact that most of them are actually grammatically structured in the same way whether is it hate speech or not.

### Models with best parameters


```python
def print_results(model, pred):

    tn, fp, fn, tp = confusion_matrix(y_test, model.predict(X_test_cvec)).ravel()
    print(f"{str(model).split('(')[0]} Confusion Matrix:")
    print(f"True Negatives: {tn}")
    print(f"False Positives: {fp}")
    print(f"False Negatives: {fn}")
    print(f"True Positives: {tp}")
    print('\n')

    report = classification_report(y_test, pred, target_names=['Predict 0', 'Predict 1'], output_dict=True)
    class_table = pd.DataFrame(report).transpose()
    display(class_table)
```


```python
X=df['tok_lemma']
y=df['hate']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=28)
```

#### Logistic Regression


```python
cvec = CountVectorizer(max_df=0.95, max_features=10000, min_df=3, ngram_range=(1,1))
logreg = LogisticRegression(C=1.0)

X_train_cvec = cvec.fit_transform(X_train)
X_test_cvec = cvec.transform(X_test)
logreg.fit(X_train_cvec, y_train)
pred_lr = logreg.predict(X_test_cvec)
```

    /Users/clementow/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)



```python
print_results(logreg, pred_lr)
```

    LogisticRegression Confusion Matrix:
    True Negatives: 7183
    False Positives: 503
    False Negatives: 920
    True Positives: 3984





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Predict 0</td>
      <td>0.886462</td>
      <td>0.934556</td>
      <td>0.909874</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <td>Predict 1</td>
      <td>0.887898</td>
      <td>0.812398</td>
      <td>0.848472</td>
      <td>4904.000000</td>
    </tr>
    <tr>
      <td>accuracy</td>
      <td>0.886974</td>
      <td>0.886974</td>
      <td>0.886974</td>
      <td>0.886974</td>
    </tr>
    <tr>
      <td>macro avg</td>
      <td>0.887180</td>
      <td>0.873477</td>
      <td>0.879173</td>
      <td>12590.000000</td>
    </tr>
    <tr>
      <td>weighted avg</td>
      <td>0.887021</td>
      <td>0.886974</td>
      <td>0.885957</td>
      <td>12590.000000</td>
    </tr>
  </tbody>
</table>
</div>


#### Logistic Regression with balanced classes


```python
def get_class_weights(y):
    majority = max(y.value_counts())
    return  {cls: float(majority/count) for cls, count in enumerate(y.value_counts())}

class_weights = get_class_weights(y)
class_weights
```




    {0: 1.0, 1: 1.5673209278613307}




```python
cvec = CountVectorizer(max_df=0.95, max_features=10000, min_df=3, ngram_range=(1,1))
logreg_bal = LogisticRegression(C=1.0, class_weight=class_weights)

X_train_cvec = cvec.fit_transform(X_train)
X_test_cvec = cvec.transform(X_test)
logreg_bal.fit(X_train_cvec, y_train)
pred_lr_bal = logreg_bal.predict(X_test_cvec)
```

    /Users/clementow/opt/anaconda3/lib/python3.7/site-packages/sklearn/linear_model/logistic.py:432: FutureWarning: Default solver will be changed to 'lbfgs' in 0.22. Specify a solver to silence this warning.
      FutureWarning)



```python
print_results(logreg_bal, pred_lr_bal)
```

    LogisticRegression Confusion Matrix:
    True Negatives: 7063
    False Positives: 623
    False Negatives: 845
    True Positives: 4059





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Predict 0</td>
      <td>0.893146</td>
      <td>0.918944</td>
      <td>0.905861</td>
      <td>7686.0000</td>
    </tr>
    <tr>
      <td>Predict 1</td>
      <td>0.866937</td>
      <td>0.827692</td>
      <td>0.846860</td>
      <td>4904.0000</td>
    </tr>
    <tr>
      <td>accuracy</td>
      <td>0.883400</td>
      <td>0.883400</td>
      <td>0.883400</td>
      <td>0.8834</td>
    </tr>
    <tr>
      <td>macro avg</td>
      <td>0.880042</td>
      <td>0.873318</td>
      <td>0.876361</td>
      <td>12590.0000</td>
    </tr>
    <tr>
      <td>weighted avg</td>
      <td>0.882937</td>
      <td>0.883400</td>
      <td>0.882879</td>
      <td>12590.0000</td>
    </tr>
  </tbody>
</table>
</div>


#### Best Model Comparison Summary

|                                         | F1 score | Recall |
|-----------------------------------------|----------|--------|
| Logistic Regression                     | 87.91%   | 87.35% |
| Logistic Regression with class weights | 87.64%   | 87.33% |

Logistic Regression without class weights is the best performing one with an F1 score of `87.91%` and the best recall.

#### Best Model Intepretation


```python
coefs=logreg.coef_[0]

word_coef = pd.DataFrame({'word': cvec.get_feature_names() , 'coeff': np.exp(coefs)})

print("Top 50 Features for Logistic Regression & CVec on Hate = 1")
word_coef.sort_values(by='coeff' , ascending=False).head(50)
```

    Top 50 Features for Logistic Regression & CVec on Hate = 1





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>6017</td>
      <td>nigger</td>
      <td>376.591530</td>
    </tr>
    <tr>
      <td>3113</td>
      <td>faggot</td>
      <td>363.563494</td>
    </tr>
    <tr>
      <td>7510</td>
      <td>retard</td>
      <td>161.153371</td>
    </tr>
    <tr>
      <td>7512</td>
      <td>retarded</td>
      <td>139.822428</td>
    </tr>
    <tr>
      <td>2057</td>
      <td>cunt</td>
      <td>100.222272</td>
    </tr>
    <tr>
      <td>8388</td>
      <td>spic</td>
      <td>85.527322</td>
    </tr>
    <tr>
      <td>5872</td>
      <td>muzzie</td>
      <td>79.058264</td>
    </tr>
    <tr>
      <td>9730</td>
      <td>wetback</td>
      <td>72.413391</td>
    </tr>
    <tr>
      <td>3111</td>
      <td>fag</td>
      <td>66.935363</td>
    </tr>
    <tr>
      <td>2701</td>
      <td>dyke</td>
      <td>45.077296</td>
    </tr>
    <tr>
      <td>7511</td>
      <td>retardation</td>
      <td>35.366813</td>
    </tr>
    <tr>
      <td>4879</td>
      <td>kike</td>
      <td>33.158387</td>
    </tr>
    <tr>
      <td>9230</td>
      <td>twat</td>
      <td>32.439435</td>
    </tr>
    <tr>
      <td>9801</td>
      <td>wigger</td>
      <td>29.150413</td>
    </tr>
    <tr>
      <td>3114</td>
      <td>faggotry</td>
      <td>25.496369</td>
    </tr>
    <tr>
      <td>7513</td>
      <td>retards</td>
      <td>23.443762</td>
    </tr>
    <tr>
      <td>7182</td>
      <td>raghead</td>
      <td>22.446466</td>
    </tr>
    <tr>
      <td>6018</td>
      <td>niggers</td>
      <td>16.405627</td>
    </tr>
    <tr>
      <td>9110</td>
      <td>tranny</td>
      <td>15.947645</td>
    </tr>
    <tr>
      <td>5119</td>
      <td>libtard</td>
      <td>12.504445</td>
    </tr>
    <tr>
      <td>2061</td>
      <td>cunty</td>
      <td>11.256686</td>
    </tr>
    <tr>
      <td>4123</td>
      <td>homos</td>
      <td>9.526618</td>
    </tr>
    <tr>
      <td>3117</td>
      <td>faggy</td>
      <td>9.053693</td>
    </tr>
    <tr>
      <td>9548</td>
      <td>vietnam</td>
      <td>8.863271</td>
    </tr>
    <tr>
      <td>598</td>
      <td>autism</td>
      <td>7.939513</td>
    </tr>
    <tr>
      <td>4119</td>
      <td>homo</td>
      <td>7.409377</td>
    </tr>
    <tr>
      <td>5818</td>
      <td>mudshark</td>
      <td>7.393040</td>
    </tr>
    <tr>
      <td>6010</td>
      <td>nig</td>
      <td>7.299513</td>
    </tr>
    <tr>
      <td>7112</td>
      <td>pussyboy</td>
      <td>6.797214</td>
    </tr>
    <tr>
      <td>2058</td>
      <td>cuntfuse</td>
      <td>6.649853</td>
    </tr>
    <tr>
      <td>8205</td>
      <td>slut</td>
      <td>6.633148</td>
    </tr>
    <tr>
      <td>2059</td>
      <td>cuntish</td>
      <td>6.316475</td>
    </tr>
    <tr>
      <td>5860</td>
      <td>mussolini</td>
      <td>6.211510</td>
    </tr>
    <tr>
      <td>6015</td>
      <td>nigga</td>
      <td>5.953319</td>
    </tr>
    <tr>
      <td>1442</td>
      <td>chinaman</td>
      <td>5.951831</td>
    </tr>
    <tr>
      <td>2060</td>
      <td>cunts</td>
      <td>5.916409</td>
    </tr>
    <tr>
      <td>5780</td>
      <td>moslem</td>
      <td>5.701641</td>
    </tr>
    <tr>
      <td>2939</td>
      <td>esque</td>
      <td>5.655120</td>
    </tr>
    <tr>
      <td>3886</td>
      <td>halfwit</td>
      <td>5.635636</td>
    </tr>
    <tr>
      <td>298</td>
      <td>americunt</td>
      <td>5.583840</td>
    </tr>
    <tr>
      <td>7111</td>
      <td>pussy</td>
      <td>5.564052</td>
    </tr>
    <tr>
      <td>8984</td>
      <td>thundercunt</td>
      <td>5.550721</td>
    </tr>
    <tr>
      <td>9772</td>
      <td>whitey</td>
      <td>5.458723</td>
    </tr>
    <tr>
      <td>6016</td>
      <td>niggas</td>
      <td>5.407924</td>
    </tr>
    <tr>
      <td>5826</td>
      <td>mulatto</td>
      <td>5.371978</td>
    </tr>
    <tr>
      <td>7948</td>
      <td>sewage</td>
      <td>5.210900</td>
    </tr>
    <tr>
      <td>4091</td>
      <td>hoe</td>
      <td>5.181927</td>
    </tr>
    <tr>
      <td>878</td>
      <td>bitch</td>
      <td>5.125506</td>
    </tr>
    <tr>
      <td>7321</td>
      <td>redneck</td>
      <td>4.943599</td>
    </tr>
    <tr>
      <td>8808</td>
      <td>tard</td>
      <td>4.932617</td>
    </tr>
  </tbody>
</table>
</div>



The model is learning that many different offensive words that are actually contributing to hate speech. This is the reason why Logistic Regression has been the go to classifier for so many years with reasonable performances.

But of course, ideally to reach above 90% perhaps a context based classifier might be useful to reduce the False Negatives.

The reason why unigram for the CountVectorizer is most useful for the classifier to make a decision is likely because of the fact that most offensive words come in a single word which are highly probable to be hate speech related.  


```python
print("Top 20 Features for Logistic Regression & CVec on Hate = 0")
word_coef.sort_values(by='coeff' , ascending=True).head(20)
```

    Top 20 Features for Logistic Regression & CVec on Hate = 0





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>word</th>
      <th>coeff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>9475</td>
      <td>van</td>
      <td>0.102721</td>
    </tr>
    <tr>
      <td>952</td>
      <td>boat</td>
      <td>0.133900</td>
    </tr>
    <tr>
      <td>6277</td>
      <td>organisation</td>
      <td>0.147655</td>
    </tr>
    <tr>
      <td>2350</td>
      <td>detail</td>
      <td>0.172020</td>
    </tr>
    <tr>
      <td>5363</td>
      <td>manipulate</td>
      <td>0.175165</td>
    </tr>
    <tr>
      <td>1288</td>
      <td>carter</td>
      <td>0.183946</td>
    </tr>
    <tr>
      <td>1187</td>
      <td>butch</td>
      <td>0.200971</td>
    </tr>
    <tr>
      <td>6476</td>
      <td>pastor</td>
      <td>0.203492</td>
    </tr>
    <tr>
      <td>2916</td>
      <td>er</td>
      <td>0.207313</td>
    </tr>
    <tr>
      <td>128</td>
      <td>adoption</td>
      <td>0.207375</td>
    </tr>
    <tr>
      <td>4781</td>
      <td>judaism</td>
      <td>0.238692</td>
    </tr>
    <tr>
      <td>9350</td>
      <td>unhinged</td>
      <td>0.241239</td>
    </tr>
    <tr>
      <td>1471</td>
      <td>chuckle</td>
      <td>0.241812</td>
    </tr>
    <tr>
      <td>1710</td>
      <td>complicit</td>
      <td>0.242037</td>
    </tr>
    <tr>
      <td>4779</td>
      <td>jt</td>
      <td>0.249178</td>
    </tr>
    <tr>
      <td>6789</td>
      <td>porno</td>
      <td>0.251294</td>
    </tr>
    <tr>
      <td>8384</td>
      <td>spew</td>
      <td>0.254795</td>
    </tr>
    <tr>
      <td>2212</td>
      <td>defender</td>
      <td>0.258314</td>
    </tr>
    <tr>
      <td>5308</td>
      <td>madness</td>
      <td>0.258361</td>
    </tr>
    <tr>
      <td>531</td>
      <td>assist</td>
      <td>0.258451</td>
    </tr>
  </tbody>
</table>
</div>



For comments that contain unigrams that are not classifed as hate speech, they are usually non-offensive. Of course, given more context, they may or may not be hate speech related.

### Best Model Misclassifications


```python
# Create figure for distribution graph
plt.figure(figsize = (10,7))

# Creatinfg two histograms of observations, with blue (left) from nonhate and yellow (right) from hate
plt.hist(pred_df[pred_df['actual'] == 0]['pred_probs'],
         bins=25,
         color='b',
         alpha = 0.6,
         label='Outcome = 0 (No Hate)')
plt.hist(pred_df[pred_df['actual'] == 1]['pred_probs'],
         bins=25,
         color='orange',
         alpha = 0.6,
         label='Outcome = 1 (Hate)')

# Labeling of axes.
plt.title('Distribution of P(Outcome = 1)', fontsize=20)
plt.ylabel('Frequency', fontsize=18)
plt.xlabel('Predicted Probability that Outcome = 1', fontsize=18)

# Creating of legends
plt.legend(fontsize=20);
```


![png](Hate-speech-detection-blogpost_files/Hate-speech-detection-blogpost_73_0.png)



```python
# Create figure for distribution graph
plt.figure(figsize = (10,7))

# Creatinfg two histograms of observations, with blue (left) from nonhate and yellow (right) from hate
plt.hist(pred_df[pred_df['actual'] == 0]['pred_probs'],
         bins=25,
         color='b',
         alpha = 0.6,
         label='Outcome = 0 (No Hate)')
plt.hist(pred_df[pred_df['actual'] == 1]['pred_probs'],
         bins=25,
         color='orange',
         alpha = 0.6,
         label='Outcome = 1 (Hate)')

# Labeling of axes.
plt.title('Distribution of P(Outcome = 1)', fontsize=20)
plt.ylabel('Frequency', fontsize=18)
plt.xlabel('Predicted Probability that Outcome = 1', fontsize=18)

# Creating of legends
plt.legend(fontsize=20);
```


![png](Hate-speech-detection-blogpost_files/Hate-speech-detection-blogpost_74_0.png)



```python
print_results(logreg, pred_lr)
```

    LogisticRegression Confusion Matrix:
    True Negatives: 7183
    False Positives: 503
    False Negatives: 920
    True Positives: 3984





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>precision</th>
      <th>recall</th>
      <th>f1-score</th>
      <th>support</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Predict 0</td>
      <td>0.886462</td>
      <td>0.934556</td>
      <td>0.909874</td>
      <td>7686.000000</td>
    </tr>
    <tr>
      <td>Predict 1</td>
      <td>0.887898</td>
      <td>0.812398</td>
      <td>0.848472</td>
      <td>4904.000000</td>
    </tr>
    <tr>
      <td>accuracy</td>
      <td>0.886974</td>
      <td>0.886974</td>
      <td>0.886974</td>
      <td>0.886974</td>
    </tr>
    <tr>
      <td>macro avg</td>
      <td>0.887180</td>
      <td>0.873477</td>
      <td>0.879173</td>
      <td>12590.000000</td>
    </tr>
    <tr>
      <td>weighted avg</td>
      <td>0.887021</td>
      <td>0.886974</td>
      <td>0.885957</td>
      <td>12590.000000</td>
    </tr>
  </tbody>
</table>
</div>


#### Analysis of False Negatives

<span style="color:maroon">Warning: There is presence of derogatory comments in this section and is discussed for academic and research purposes. Please skip to the subsequent sections if you find this offensive.</span>


```python
#sampling subset of 10 false negatives
lr_fn_df = pd.read_csv('../assets/logreg_fn.csv')
for i in np.random.choice(lr_fn_df.index, 5):
    display(lr_fn_df.loc[i,'comment'])
    print('\nResponse:')
    print(lr_fn_df.loc[i,'response'])
    print('\n')
    print(f"Probability: {lr_fn_df.loc[i,'pred_probs']}")
    print('----'*10)
```


    "I shouldn't even have to use the /s, it is incredibly obvious that I am being sarcastic either way."



    Response:
    ['This thread is closed. We ask that you all review our user policies and refrain from hate speech of any kind. This is a final warning. Ty.', 'Its inappropriate to use words that directly attack one based on their sexual preferences (incel and faggot) and gender (cunt).', 'Do not discuss hate speech by using hate speech there are ways to avoid this. Name calling also does not add to the conversation.']


    Probability: 0.0933714693538914
    ----------------------------------------



    'I don’t know the full details of the IMC shit. I was bored with the Kraut stupidity the first time round so this round holds little interest for me. The name wild goose however I do know. He is a legitimately horrible individual who is butt buddies with Ralph and the other IBS cunts. I believe he was the one who doxed Sargon and celebrated the suicide of the sister of a GG’er back in the day.   Honestly it’s not what he changed his mind to that makes me distrust him. It’s the degree to which he flipped and the time it took to happen. I don’t give him any credibility and doubt every thing he says and does.   You do you mate. Just remember that someone’s actions belie the truth of their words. '



    Response:
    ['Use of the c-word here is not acceptable because it is hurtful and offensive as it denigrates women.', 'Using the term "c---" pejoratively is offensive and should be avoided.']


    Probability: 0.007523960445078103
    ----------------------------------------



    ".... they're the 'dumbest of the dumb' who believe the insane Left and try to prove they're NOT against being killed by religious imbeciles !"



    Response:
    ['Use of the r-word is unacceptable in discourse as it demeans and insults people with mental disabilities.', 'I like Cindy Lauper and her song Girls Just Want to Have Fun', 'Please avoid expressions that denigrate women or people of other religions.']


    Probability: 0.06008171222560287
    ----------------------------------------



    "Sold yourself out dbag. Throwing buzzwords around instead of thinking makes you a inept yet dangerous person. Seek help, there's no shame in mental illness. "



    Response:
    ["Next we need a law where if you're caught over and over lying about these things you go to jail.", "There's no need for that language. This is a warning.", 'Stop using a medical condition as an insult.']


    Probability: 0.11629146743911055
    ----------------------------------------



    "She's a big big BIG phony beyond her heritage Claims.  She was a Financial advisor for the Clinton administration.  She audited the Derivative bonds before the Real Estate bubble even ever happened. She found that it was pure garbage and junk and would destroy the Financial health of America if those type of financial vehicles were allowed to be sold.   Had She had just a smidgen of the meddle Trump has, She would have went public. Instead She was told to keep her finding quiet and was kicked up to the Big corner office. Then upgraded on a fast track to be the Senator.   I knew She was the Shittiest person to be entrusted with creating the CFPB. The verbiage in the legalese of most Financial contracts are more oppressive than ever. Companies can legally fuck you over 10 different ways to Sunday. As long as they mention in their contract to read the terms and condition that may be in other supplemental materials like on a Website, or even have to call or email for the hard copy of those terms. The terms don't even have to be in the fine print on those contracts any more. Just be available somewhere in the universe. But make no mistake the terms and conditions have never been less transparent in the History of Mankind trading goods and services for a currency.   This Bitch is the most Vile and Evil Cunt to ever have existed.  She created more loopholes to fuck over the most vulnerable and week and people don't even realize it. They think the CFPB is there to protect them. It's there to create legal ways to fuck you over. "



    Response:
    ["Name calling the people you disagree with isn't going to help solve anything.", 'Removing the foul language will help others to understand your point of view more.', "While an argument can be made on whether certain derogatory terms are applied evenly, it doesn't help to aggressively label someone that way"]


    Probability: 0.01147579121017993
    ----------------------------------------


The false negatives can be segregated into the following categories highlighted in bold.  

__Subjective__

>'I don’t know the full details of the IMC shit. I was bored with the Kraut stupidity the first time round so this round holds little interest for me. The name wild goose however I do know. He is a legitimately horrible individual who is butt buddies with Ralph and the other IBS c\*\*\*\*. I believe he was the one who doxed Sargon and celebrated the suicide of the sister of a GG’er back in the day.   Honestly it’s not what he changed his mind to that makes me distrust him. It’s the degree to which he flipped and the time it took to happen. I don’t give him any credibility and doubt every thing he says and does.   You do you mate. Just remember that someone’s actions belie the truth of their words. '

What is hate speech is subjective and some of the human labellers might be more offended at certain swear words which are deemed derogatory whereas some might find it more acceptable.

__Mispelled derogatory terms or joined together__
> 'A f\*\*\*ing life sentence? Jeez, give that lady her p\*\*\*y pass back.'

> "Hell if you’re over here it’s the only word, damn thing fits in just about ever c\*\*\*\*ing sentence you can come up with."

>'The rest of his life in Pakistan? Not so sure that’s a preferable option to being treated like a hero as part of a Moslim gang in some British jail.'

Sometimes the words are joined together or seperated deliberately or accidentally. Since misspellings are not very common, the model does not recognise them as hate speech related.

__Dubious ones__
>"Cool, now we know it's false because you said it.  :D"

>"Half these clowns on here can't even spell philosophy without the spell checker!"

The above ones do not seem to be hate speech in nature and might be due to mislabelling.

__Contextual to the conversation__
> 'Exacly my point, and thats why we have the second amendment so if any of those monkeys try and give me consequences for my speech i can blow them away'

> "Being white just means your better than anyone else and should go the extra length for all the dumb brown people in the world(SJW logic)  Based on various garbage articles I've seen online(vox buzzfeed salon etc)  Black people can't figure out how to be places okkn time so we need to be patient with them. Black people can't figure out basic manners and politeness. Black people are scared of milk. Black peop e need their own little safe spaces so e don't spook them. Black people can't find jobs and should live off the table scraps of white people.  I could go on and on. If you frame it this way, it's pretty clear SJW's are convijnced white people are superior in every way, and shouldn't be held to the same standard. Being black is a mentalcretardation(and any other skin color to)"

> ' Wait until Canadian  women’s muslim owners are fucking them at will and then selling them to the highest bidder!   They will wish they had moved to the United States where a president would protect them! '

> "Sold yourself out dbag. Throwing buzzwords around instead of thinking makes you a inept yet dangerous person. Seek help, there's no shame in mental illness. "

And of course, there will be some comments that do not contain any of the derogatory words and yet are still deemed as hate speech with more context of the conversation.

#### Analysis of False Positives

<span style="color:maroon">Warning: There is presence of derogatory comments in this section and is discussed for academic and research purposes. Please skip to the subsequent sections if you find this offensive.</span>


```python
#sampling subset of 10 false postives
lr_fp_df = pd.read_csv('../assets/logreg_fp.csv')
for i in np.random.choice(lr_fp_df.index, 10):
    display(lr_fp_df.loc[i,'comment'])
    print('\n')
    print(f"Probability: {lr_fp_df.loc[i,'pred_probs']}")
    print('----'*10)
```


    'Point taken.  When I see gang rape, I think Muzzie.'




    Probability: 0.8553899900368416
    ----------------------------------------



    '>You\'re still pretending that there are radical dominionists in the wings, when the reality is that the contemporary right has become substantially more secular and in some ways much more socially liberal than it used to be.  If the contemporary right is becoming more liberal, that doesn\'t mean the contemporary right is changing - it means more people are accepting of liberal values (which the right has traditionally not been). And I fully agree.   But, I honestly think you are trying to paint a picture of a new, improved, updated right-wing - accepting of gays, accepting of personal freedoms when it comes to issues such as abortion etc... which at best, is simply a thinly veneered attempt to attract "new blood" by throwing out some of the ideas that simply aren\'t going to wash in the modern world (eg - Milo being "flamboyantly gay" - while at the same time taking an anti-gay stance on just about every gay rights issue - such as gay marriage which, even though he has indulged in it himself, he claims should be "called something else".)  And at worst (in the case of the far-right) is exactly the same as the old-right has always been.  You have to admit, the so-called new, "liberal" right wing which you are trying to convince me exists, and the old fashioned Bible-thumping, cross-burning right-wing have an awful lot in common - Love of Trump, hatred of feminism, hatred of Islam, rejection of #metoo, belief in traditional gender roles, willingness to engage in conspiracy theories (Pizza Gate, Sandy Hook being a false flag) and, as we\'re discussing this on a Gamergate subreddit, hatred of Anita Sarkeesian/Zoe Quinn.   And if we include Breitbart readers/commenters (who certainly would include themselves, and I don\'t really think you can exclude them simply because they don\'t fit your idea of what constitutes the "new right"), then we can include white nationalism in the mix, too.  Breitbart, by the way, to some extent, represents the more moderate end of the "far right" - I could easily have cited Andrew Anglin or Richard Spencer - outright white nationalists.    You may say they are different, but they are unequivocally the same in many ways. So it\'s really just a PR exercise.   '




    Probability: 0.7350872438669452
    ----------------------------------------



    'That is a fair concern. However, I am a hillbilly stuck in Denver. Everytime I hear one of these harpies try to to act like we are evil men because we are white and straight, I feel like reminding them just how dangerous we really are. How easy would it be for you and your buddies to leave many leftist hats on the ground? I know it would not be a challenge on my end.'




    Probability: 0.7547324069011335
    ----------------------------------------



    "There are a lot of women who are fucking cunts too, but I still love the women in my life and I know there are a majority out there who aren't dumb cunts. It doesn't mean all women have to apologize for the cunts out there. Fucking shit man, fuck progressives."




    Probability: 0.9999936816613868
    ----------------------------------------



    'Is this something that YouTube was actually planning on doing? I ask because the source was Keemstar, who no one should have as a source for anything other than how to be a total cunt. '




    Probability: 0.6660993244536081
    ----------------------------------------



    'It reminds me of the "Redneck Revolt" contingent of Antifa. They should probably change their name to Redneck Surrender.'




    Probability: 0.8374726632394158
    ----------------------------------------



    'The first three episodes were golden, I was blown away. SBC is far better doing this kind of comedy than scripted film roles. I\'ve been a fan of his since the 11 o\'clock show, and he\'s been receiving the same criticisms since those days - he\'s remained remarkably unfazed.  The show ran out of steam though. The latter half of the season was by no means terrible, but it peaked in those first three episodes. Getting that state senator to walk around saying "nigger", getting those guys dressed up as teenage girls for the Mexican coming out party, opening a brand new state of the art mosque - these were all up there with some of the best work SBC has ever done. There was really nowhere left to go after those bits, they set the bar too high too early. '




    Probability: 0.6289210491449319
    ----------------------------------------



    'Well i definitely agree with the controller being superior in some cases, I for example prefer using it in dark souls and some other 2D platformers. Just that this case was a shooter, FPS, on PC and she was saying that controller was better wtf are you using M+KB and that IMO is beyond retarded. '




    Probability: 0.7490682692133128
    ----------------------------------------



    '> The people who get called racist, xenophobic or "right-wingers" tend to be the ones who conflate all Muslims with radical Islamists.  Ayaan Hirsi Ali, Maajid Nawaz and even *Tommy Fucking Robinson* frequently make clear they aren\'t talking about all Muslims, and they frequently distinguish between "Islamists" and "most Muslims in the west." They still get called "radical racist xenophobic right-wingers."  >The experience you will gain, backed with factual statistics, will show you that the vast majority of Muslims are peaceful and simply want to be left alone to get on with their lives.   The factual statistics also suggest that a very large percentage of Muslims are socially and theologically conservative, and would favor laws that restrict our civil liberties in the name of their religion. A survey of British Muslims found that a *supermajority* thought homosexuality should be criminalized.  Sure, they aren\'t necessarily *jihadists*. But if you believe in forcing society to live by Islamic norms *even via the ballot box* then you\'re an Islamist ("Jihadists" are Islamists who support terrorism as a means to forcing society to live by Islamic norms).   Yes, there are many Muslims whom are not Islamists. But we need to take the problem seriously. We saw the theocratic nonsense spouted by the religious right back in the George W Bush administration for the *threat it was*, and we didn\'t make excuses for them like "but they\'re nonviolent, they want a democratic process to restrict our rights." Islamists should be viewed with the same suspicion, if not more, that was cast upon the Dominionists.  >When people start making claims that "all" Muslims are jihadists, or being Muslim inherently means you are a violent extremist, or follow an extremist ideology - that\'s when you will get called racist, xenophobic or right-wing.  Again, not even Tommy Robinson supports that viewpoint. In addition, some people are *very eager to conflate* the proposition that "some verses in the Quran and some theological positions that are prominent in the Islamic world logically support Jihadists" with the proposition that "all Muslims are violent extremists." Take a look at how Sam Harris was treated by Ben Affleck.   >Do you have a list of "official" organs of the "establishment left"? I didn\'t know there was such a thing.  Think "major, center-left social-democratic political parties," left-leaning MSM outlets, and most academics. '




    Probability: 0.9918611748424442
    ----------------------------------------



    'Frankly I think it’s a good thing that racists have to hide their disgusting inhumane views, but sadly that’s changing. We’ve got these idiots on TiA, and cunts with tiki torches and cargo shorts on TV. Mental.  '




    Probability: 0.9392706010700892
    ----------------------------------------


The false positives can be segregated into the following categories highlighted in bold.

__Mislabelled__

>'Point taken.  When I see gang rape, I think Muzzie.'

>'Frankly I think it’s a good thing that racists have to hide their disgusting inhumane views, but sadly that’s changing. We’ve got these idiots on TiA, and c\*\*\*\* with tiki torches and cargo shorts on TV. Mental.  '

>'Is this something that YouTube was actually planning on doing? I ask because the source was Keemstar, who no one should have as a source for anything other than how to be a total c\*\*\*. '

>'It reminds me of the "Redneck Revolt" contingent of Antifa. They should probably change their name to Redneck Surrender.'

The above comments are definitely hate speech and the model did well by being sensitive to such derogatory words and to detect mislabelled comments.  

__Sensitive to strong words__

>'Well i definitely agree with the controller being superior in some cases, I for example prefer using it in dark souls and some other 2D platformers. Just that this case was a shooter, FPS, on PC and she was saying that controller was better wtf are you using M+KB and that IMO is beyond retarded. '

>'The first three episodes were golden, I was blown away. SBC is far better doing this kind of comedy than scripted film roles. I\'ve been a fan of his since the 11 o\'clock show, and he\'s been receiving the same criticisms since those days - he\'s remained remarkably unfazed.  The show ran out of steam though. The latter half of the season was by no means terrible, but it peaked in those first three episodes. Getting that state senator to walk around saying "n\*\*\*\*\*", getting those guys dressed up as teenage girls for the Mexican coming out party, opening a brand new state of the art mosque - these were all up there with some of the best work SBC has ever done. There was really nowhere left to go after those bits, they set the bar too high too early. '

>'That is a fair concern. However, I am a hillbilly stuck in Denver. Everytime I hear one of these harpies try to to act like we are evil men because we are white and straight, I feel like reminding them just how dangerous we really are. How easy would it be for you and your buddies to leave many leftist hats on the ground? I know it would not be a challenge on my end.'

Generally the model is sensitive to derogatory terms and we have trained it to be so to catch more and lower the false negatives.

"Hillbilly" is a considered a derogatory term in America for people who live in the countryside. However, it is not considered hate speech as he or she is directing it at himself and not at other people, thereby not satisfying the hate speech definition.

Hence, more context will be needed to really acertain if it is indeed hate speech or not.


__Many top features of the model in one comment__

>"There are a lot of women who are f\*\*\*ing c\*\*\*\* too, but I still love the women in my life and I know there are a majority out there who aren't dumb c\*\*\*\*. It doesn't mean all women have to apologize for the c\*\*\*\* out there. F\*\*\*ing shit man, f\*\*\* progressives."

> '> The people who get called racist, xenophobic or "right-wingers" tend to be the ones who conflate all Muslims with radical Islamists.  Ayaan Hirsi Ali, Maajid Nawaz and even *Tommy Fucking Robinson* frequently make clear they aren\'t talking about all Muslims, and they frequently distinguish between "Islamists" and "most Muslims in the west." They still get called "radical racist xenophobic right-wingers."  >The experience you will gain, backed with factual statistics, will show you that the vast majority of Muslims are peaceful and simply want to be left alone to get on with their lives.   The factual statistics also suggest that a very large percentage of Muslims are socially and theologically conservative, and would favor laws that restrict our civil liberties in the name of their religion. A survey of British Muslims found that a *supermajority* thought homosexuality should be criminalized.  Sure, they aren\'t necessarily *jihadists*. But if you believe in forcing society to live by Islamic norms *even via the ballot box* then you\'re an Islamist ("Jihadists" are Islamists who support terrorism as a means to forcing society to live by Islamic norms).   Yes, there are many Muslims whom are not Islamists. But we need to take the problem seriously. We saw the theocratic nonsense spouted by the religious right back in the George W Bush administration for the *threat it was*, and we didn\'t make excuses for them like "but they\'re nonviolent, they want a democratic process to restrict our rights." Islamists should be viewed with the same suspicion, if not more, that was cast upon the Dominionists.  >When people start making claims that "all" Muslims are jihadists, or being Muslim inherently means you are a violent extremist, or follow an extremist ideology - that\'s when you will get called racist, xenophobic or right-wing.  Again, not even Tommy Robinson supports that viewpoint. In addition, some people are *very eager to conflate* the proposition that "some verses in the Quran and some theological positions that are prominent in the Islamic world logically support Jihadists" with the proposition that "all Muslims are violent extremists." Take a look at how Sam Harris was treated by Ben Affleck.   >Do you have a list of "official" organs of the "establishment left"? I didn\'t know there was such a thing.  Think "major, center-left social-democratic political parties," left-leaning MSM outlets, and most academics. '

Due to the sensitivity of the model, whenever there are many of such words in a comment, it is very highly likely a hate speech comment. However, the first comment might not be really hate speech but it could be subjective and offensive for some.

Overall, the model did well in predicting if the comments are hate speech or not. However, many of the mislabellings can be avoided with more context as it is one of the hot topics of the NLP space.

# Limitations and Future Work

The challenge faced by automatic hate speech detection is the subjectivity of whether a comment is considered hate speech or not. This can be better managed by having more people labelling these datasets to cross reference and to take a majority vote.

Another challenge is that many new urban words that are deemed derogatory are coined every few years or decades and the models that are developed now might be obsolete in the future. Constant training of new data sets will thus be paramount in overriding this problem.

As with any hate speech classification problem, context is needed to determine whether it is hate speech or not in many cases. Looking at the context of the text of how a word is being used and linguistic features will be a better way of understanding text. Of course, understanding sarcasm is one of the ongoing research which will help immensely in NLP tasks and higher accuracy rates. Therefore, more models have to be developed to train on learning to read context left or right of the target word or having "multiple views" of the same comment by using Multi-view ensemble stacking models.

# Conclusions

With the rise of social media and users being able to stay anonymous, hate speech detection is ever important in the digital age.

We present current approaches to this classification task and also explored different techniques including deep learning models and state-of-the-art models such as BERT.

| Classifier                              | F1 score | Recall |
|-----------------------------------------|----------|--------|
| Logistic Regression                     | **87.91%**   | **87.35%** |
| Logistic Regression with balanced class | 87.64%   | 87.33% |
| LSTM - word embeddings on dataset       | 85.18%   |    -   |
| LSTM & CNN - word embeddings on dataset | 84.95%   |    -   |
| LSTM 1 - pre-trained word embeddings    | 81.75%   |    -   |
| LSTM 2 - pre-trained word embeddings    | 84.36%   |    -   |
| BERT                                    | 87%      | 87%    |

Even though context is important in determining if a comment is hate speech or not, the simplest classifier, Logistic Regression, is actually the best performing one. This goes to show that at times, the simpler the classifier the better in terms of interpretability and it has made it easier to choose the best model with a superior F1 score.

In comparison with the state-of-the-art NLP BERT model, Logistic Regression was still able to perform very well while generalises well for this specific task. It is no wonder why Logistic Regression has been around for many years and continues to be widely used.
