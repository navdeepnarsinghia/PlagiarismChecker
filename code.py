# Information Retrieval Assignment-1
# Topic: Plagiarism detector
# No. of Group Members = 1
# Name: Navdeep Singh Narsinghia
# ID: 2017B5A71675H

import os
import nltk
import numpy as np
import math
import time
from nltk.corpus import stopwords


start_time = time.time()

# function to extract contents and name of file from training files
def getText(filename):
    names.append(filename)
    doc = open(filename, encoding="utf8", errors='ignore')
    fullText = []
    for line in doc:
        fullText.append(line)
    return '\n'.join(fullText)


# Prerocessing and building vocabulary of all documents
def preprocessing(trainingdata):
    lex = set()
    for doc in trainingdata:
        # word tockenization
        word_token = [word for word in doc.split()]
        lower_word_list = [i.lower() for i in word_token]

        # Porter stemming
        porter = nltk.PorterStemmer()
        stemmed_word = [porter.stem(t) for t in lower_word_list]

        # removing stop words
        stop_words = set(stopwords.words('english'))
        bag_of_words = [w for w in stemmed_word if not w in stop_words]

        lex.update(bag_of_words)
    return lex


# calculating term frequency
def termfreq(term, document):
    return freq(term, document)

# calculating term frequency
def freq(term, document):
    return document.split().count(term)


# normalizing vectors
def l2_normalize(vec):
    denomi = np.sum([ele ** 2 for ele in vec])
    return [(ele / math.sqrt(denomi)) for ele in vec]


# calculating document frequency
def numofDocs(word, doclist):
    doccount = 0
    for doc in doclist:
        if freq(word, doc) > 0:
            doccount += 1
    return doccount


# calculating idf frequency weighting
def idf(word, doclist):
    n_samples = len(doclist)
    df = numofDocs(word, doclist)
    return np.log(n_samples / (1 + df))


# building idf matrix
def idf_matrix(idf_vector):
    idf_mat = np.zeros((len(idf_vector), len(idf_vector)))
    np.fill_diagonal(idf_mat, idf_vector)
    return idf_mat

print("Hello! Welcome to my Plagiarism detector\n")  # Welcome message

docFiles = []  # stores contents of all documents
names = []  # stores names of all documents

# scanning through training files
os.chdir('D:\\corpus\\')
for filename in os.listdir():
    if filename.endswith('.txt'):
        filename = getText(filename)
        docFiles.append(filename)

traininglen = len(docFiles)  # total training documents

# scanning through test file
os.chdir('D:\\query\\')
for filename in os.listdir():
    if filename.endswith('.txt'):
        filename = getText(filename)
        docFiles.append(filename)

testlen = len(docFiles) - traininglen  # total queries

print("Preprocessing, please wait.\n")  # message to user

vocabulary = preprocessing(docFiles)  # preprocessing and building vocabulary

doc_term_matrix = []

# building tf vectors
for doc in docFiles:
    tf_vector = [termfreq(word, doc) for word in vocabulary]
    tf_vector_string = ', '.join(format(freq, 'd') for freq in tf_vector)
    doc_term_matrix.append(tf_vector)

print("Calculating plagiarism. Please wait..........\n")  # message to user

# normalized document term matrix
doc_term_matrix_l2 = []
for vec in doc_term_matrix:
    doc_term_matrix_l2.append(l2_normalize(vec))

# idf vector
idf_vector = [idf(word, docFiles) for word in vocabulary]

# building idf matrix
idf_matrix = idf_matrix(idf_vector)

# tf-idf document term matrix
doc_term_matrix_tfidf = []

# performing tf-idf matrix multiplication
for tf_vector in doc_term_matrix:
    doc_term_matrix_tfidf.append(np.dot(tf_vector, idf_matrix))

doc_term_matrix_tfidf_l2 = []

for tf_vector in doc_term_matrix_tfidf:
    doc_term_matrix_tfidf_l2.append(l2_normalize(tf_vector))

# cosine distance and finf=ding similarity
for i in range(traininglen, len(docFiles), 1):

    print("Plagiarism report for " + names[i])
    print("\n            DOCUMENT NAME          SIMILARITY")

    finalname = []  # stores names of similar documents
    finalsim = []  # stores amount of similarity

    for j in range(traininglen):
        answer = nltk.cluster.util.cosine_distance(doc_term_matrix_tfidf_l2[i],
                                                   doc_term_matrix_tfidf_l2[j])  # cosine distance calculation
        cos_sim = 1 - answer

        plagiarism = int(cos_sim * 100)  # percenage of similarity

        finalname.append(names[j])
        finalsim.append(plagiarism)

    finalname = [x for _, x in sorted(zip(finalsim, finalname))]  # sorting documents by amount of similarity
    finalsim.sort(reverse=True)  # ordering in decreasing order of similarity
    finalname.reverse()

    for k in range(traininglen):
        print("%25s" % finalname[k] + "          " + "%s %%" % finalsim[k])

    print("\n")

print("Time: " , time.time() - start_time, " sec\n")