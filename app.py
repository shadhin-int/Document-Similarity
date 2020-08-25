
import nltk
import gensim
nltk.download('punkt')
from nltk.tokenize import *


# data = "Mars is a cold desert world. It is half the size of Earth.."
# print(word_tokenize(data))
# print("==============")
# print(sent_tokenize(data))

file_docs = []

with open ('testfile.txt') as file:
    tokens = sent_tokenize(file.read())
    for line in tokens:
        file_docs.append(line)

print("Number of Documents:", len(file_docs))
print("============")

gen_docs = [[w.lower() for w in word_tokenize(text)] for text in file_docs]
print(gen_docs)
print("============")

dictionary = gensim.corpora.Dictionary(gen_docs)
print(dictionary.token2id)