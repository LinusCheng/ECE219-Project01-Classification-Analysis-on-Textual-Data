import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import NMF
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from nltk.stem.snowball import SnowballStemmer


def fetch_data(categories):
    return fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42), \
    fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)
    
categories = ['comp.graphics', 'comp.os.ms-windows.misc', 'comp.sys.ibm.pc.hardware', 'comp.sys.mac.hardware', 'rec.autos', 'rec.motorcycles', 'rec.sport.baseball', 'rec.sport.hockey']
twenty_train, twenty_test = fetch_data(categories)
print("Num of doc in train" ,len(twenty_train.data))
print("Num of doc in test"  ,len(twenty_test.data))
stop_words = text.ENGLISH_STOP_WORDS


"""=============================================================================================================="""
""" (a) """
print("=== Part (a) ===")


his = [0] * 8
for t in twenty_train.target:
     his[t] += 1
print(his)

x = np.arange(8)
plt.barh(x, his, alpha=0.8)
plt.yticks(x, twenty_train.target_names, color='pink')
plt.xlabel("Amount", color='pink')
plt.ylabel("Categories", color='pink')
plt.title('Number of training documents', color='pink')


"""=============================================================================================================="""
""" (b) """
print("=== Part (b) ===")
# to capture the importance of a word to a document 
# For example, if a corpus is about computer accessories then words such as “computer” “software” “purchase” 
# will be present in almost every document and their frequency is not a distinguishing feature for any document in the corpus.


def Stemmer(Data):
    for i in range(len(Data)):
        Data[i] = ' '.join(map(SnowballStemmer("english").stem, CountVectorizer().build_analyzer()(Data[i])))

def Counter(min_df, train, test):
    # The default regexp select tokens of 2 or more alphanumeric characters 
    # punctuation is completely ignored and always treated as a token separator
    print("min_df =",min_df)
    Stemmer(train.data)
    Stemmer(test.data)
    
    count_vect = CountVectorizer(min_df=min_df, stop_words=stop_words)
    X_train_counts = count_vect.fit_transform(train.data)
    XX_train  = X_train_counts.toarray()
    print("Shape of train =",XX_train.shape)
    X_test_counts = count_vect.transform(test.data)
    XX_test  = X_test_counts.toarray()
    print("Shape of test =",XX_test.shape)
    return X_train_counts, X_test_counts

X_train_counts_2, X_test_counts_2 = Counter(2, twenty_train, twenty_test)
X_train_counts_5, X_test_counts_5 = Counter(5, twenty_train, twenty_test)

# 25343
# 10664



""" TFIDF """
# idf(t) = log[n / df(t)] + 1
# n is the total number of documents
# df(t) is the document frequency; the document
# frequency is the number of documents that contain the term t



def tfidf(X_train_counts, X_test_counts):
    tfidf_transformer = TfidfTransformer()
    print(X_train_counts.shape)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    return X_train_tfidf, X_test_tfidf

X_train_tfidf_2, X_test_tfidf_2 = tfidf(X_train_counts_2, X_test_counts_2)
X_train_tfidf_5, X_test_tfidf_5 = tfidf(X_train_counts_5, X_test_counts_5)

print ("X_train_tfidf_2.shape =",X_train_tfidf_2.shape)
print ("X_test_tfidf_2.shape =",X_test_tfidf_2.shape)
print ("X_train_tfidf_5.shape =",X_train_tfidf_5.shape)
print ("X_test_tfidf_5.shape =",X_test_tfidf_5.shape)


# print("\n\n\n\n")
# print("train 2&5")
# print ('-' * 20)
# print ("X_train_counts_2.toarray")
# print (X_train_counts_2.toarray()[:30,:5])
# print ('-' * 20)
# print ("X_train_tfidf_2.toarray")
# print (X_train_tfidf_2.toarray()[:30,:5])

# print ('-' * 20)
# print ("X_train_counts_5.toarray")
# print (X_train_counts_5.toarray()[:30,:5])
# print ('-' * 20)
# print ("X_train_tfidf_5.toarray")
# print (X_train_tfidf_5.toarray()[:30,:5])



"""=============================================================================================================="""
""" (c) """
print("=== Part (c) ===")

# TFxICF(t,c) = tf(t,c) * icf(t)
# icf(t) = log [num_classes / cf(t)] + 1 
# tf(t,c) represents the term freq in a class c
#
# Find the 10 most significant terms

 
twenty_all = fetch_20newsgroups(subset='all', shuffle=True, random_state=42)

print("Num of docs =" ,len(twenty_all.data))
his = [0] * 20
for t in twenty_all.target:
     his[t] += 1
print("The doc numbers of all classes =" ,his)

Stemmer(twenty_all.data)
count_vect_2 = CountVectorizer(min_df=2, stop_words=stop_words)
all_counts_2 = count_vect_2.fit_transform(twenty_all.data)

count_vect_5 = CountVectorizer(min_df=5, stop_words=stop_words)
all_counts_5 = count_vect_5.fit_transform(twenty_all.data)

print(all_counts_2.shape, all_counts_5.shape)



""" ===  """
all_tfidf_2 = TfidfTransformer().fit_transform(all_counts_2)
all_tfidf_5 = TfidfTransformer().fit_transform(all_counts_5)

def required_class(all_tfidf, target):
    ret = np.zeros((4, all_tfidf.shape[1]))
    for i in range(all_tfidf.shape[0]):
        if target[i] == 3:
            ret[0,:] += all_tfidf[i, :]
            continue
        if target[i] == 4:
            ret[1,:] += all_tfidf[i, :]
            continue
        if target[i] == 6:
            ret[2,:] += all_tfidf[i, :]
            continue
        if target[i] == 15:
            ret[3,:] += all_tfidf[i, :]
            continue
    return ret

r_classes_2_tfidf = required_class(all_tfidf_2, twenty_all.target)
r_classes_5_tfidf = required_class(all_tfidf_5, twenty_all.target)


""" print the terms """

def pname(c, v):
    for i in range(4):
        S = sorted(range(c.shape[1]), key=lambda j: c[i][j])[-10:]
        for s in reversed(S):
            print(v.get_feature_names()[s], end=" ")
        print()
        
pname(r_classes_2_tfidf, count_vect_2)
pname(r_classes_5_tfidf, count_vect_5)








