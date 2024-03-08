import numpy as np
from vector_store import VectorStore

vector_store = VectorStore()    # Creaate a vector store instance.

# Define your sentences.
sentences = [
    "I eat mangoes",
    "Mangoe is my favorite fruit",
    "Mangoes grow in summer",
    "Mangoes are sweet",
    "Alphonso is my favorite mango"
]

# Tokenize and create vocabulary.
vocabulary = set()
for sentence in sentences:
    words = sentence.lower().split(" ")
    for word in words:
        vocabulary.add(word)

# Assign unique indices to each word in the vocabulary.
word_to_index = {word: index for index, word in enumerate(vocabulary)}

# Vectorization
sentence_vectors = {}
for sentence in sentences:
    words = sentence.lower().split(" ")
    vector = np.zeros(len(vocabulary))

    for word in words:  # ngram model
        vector[word_to_index[word]] += 1

    sentence_vectors[sentence] = vector


# Storing the vectors in the vector store
for sentence, vector in sentence_vectors.items():
    vector_store.add_vector(sentence, vector)

# Searching for similarity
query_sentence = 'Mango is the best fruit'

# Convert the query sentence into vector.
query_words = sentence.lower().split(" ")
query_vector = np.zeros(len(vocabulary))
for word in query_words:  # ngram model
    if word in vocabulary:
        query_vector[word_to_index[word]] += 1

similar_sentences = vector_store.get_similar_vectors(query_vector, 2)

for sentence, score in similar_sentences:
    print(f'retrieved sentence: {sentence}, corresponding score: {score}')