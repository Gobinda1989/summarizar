
import os
import numpy as np
import pandas as pd
import nltk
nltk.download()
from gensim.models import Word2Vec
from scipy import spatial
import networkx as nx
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words('english')


def read_text_file(file_path):
    #feature=[]
    with open(file_path, 'r',encoding="utf8") as f:
        txt=f.read()
        #feature.append(txt.replace("\n", ""))
    return txt.replace("\n", "")

def create_dataset(path):
    print("inside create dataset")
    # Change the directory
    os.chdir(path)
    feature = []
    for file in os.listdir():
        # Check whether file is in text format or not
        #if file.endswith(".txt"):
        file_path = f"{path}\\{file}"
        # call read text file function
        feature.append(read_text_file(file_path))
    print(feature[0])

    return feature

'''
This method is used to create list of sentences from the paragraph. It will clean the text also.
input- text
output- list of word tokens, sentence list

'''
def clean_data(features):
    sentences = []
    sentences=sent_tokenize(features)

    #sentences = [y for x in sentences for y in x]  # flatten list
    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ")

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]
    #sen_new = " ".join([i for i in clean_sentences if i not in stop_words])
    sentence_tokens = [[words for words in sentence.split(' ')  if words not in stop_words] for sentence in
                       clean_sentences]
    return sentence_tokens,sentences

'''
this method is for word embedding

input - sentence tokens
output- embedded tokens
'''

def create_word_embedding(sentence_tokens):
    w2v = Word2Vec(sentence_tokens, size=1, min_count=1, iter=1000)
    sentence_embeddings = [[w2v[word][0] for word in words] for words in sentence_tokens]
    max_len = max([len(tokens) for tokens in sentence_tokens])
    sentence_embeddings = [np.pad(embedding, (0, max_len - len(embedding)), 'constant') for embedding in
                           sentence_embeddings]
    return sentence_embeddings

'''
This method is used to create cosine similarit matrix
input- sentence tokens , sentence embedding
output- similarity matrix
'''

def create_similarity_matrix(sentence_tokens,sentence_embeddings):
    similarity_matrix = np.zeros([len(sentence_tokens), len(sentence_tokens)])
    for i, row_embedding in enumerate(sentence_embeddings):
        for j, column_embedding in enumerate(sentence_embeddings):
            similarity_matrix[i][j] = 1 - spatial.distance.cosine(row_embedding, column_embedding)
    return similarity_matrix




if __name__ == "__main__" :
    print("inside main")
    cwd = os.getcwd()
    path=cwd+'\\dataset\\stories_text_summarization_dataset_test'
    feature_list = create_dataset(path)
    extract_list=[]
    for features in feature_list:
        print(features)
        sentence_tokens ,sentences= clean_data(features)
        print(sentences)
        sentence_embeddings = create_word_embedding(sentence_tokens)
        similarity_matrix=create_similarity_matrix(sentence_tokens, sentence_embeddings)
        ''' Convert the similarity matrix to a network/graph and apply pagerank '''
        nx_graph = nx.from_numpy_array(similarity_matrix)
        scores = nx.pagerank(nx_graph)
        ''' taking top 4 sentence as extracted summar '''
        top_sentence = {sentence: scores[index] for index, sentence in enumerate(sentences)}
        top = dict(sorted(top_sentence.items(), key=lambda x: x[1], reverse=True)[:4])
        print(top)
        sent_list=[]
        for sent in sentences:
            if sent in top.keys():
                print(sent)
                sent_list.append(sent)
        extract_list.append(' '.join([str(elem) for elem in sent_list]))
    print(extract_list[0])
    df=pd.DataFrame({"Actual Document": feature_list,"Extracted Document":extract_list})
    df.to_csv('extract.csv',index=False)
    print("***********************completed**********************************")



