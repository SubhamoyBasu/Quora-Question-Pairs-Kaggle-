
# coding: utf-8

# # Quora Question Pairs (Kaggle)

# In[1]:


import numpy as np
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

MAX_NB_WORDS = 200000
MAX_SEQUENCE_LENGTH = 25
EMBEDDING_DIM = 300


# ### Load Data

# In[2]:


train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")


# ### Clean Data

# In[3]:


# Clean text
train_data['q1_processed'] = train_data['question1'].apply(lambda x: re.sub(r"what's", "what is ",str(x)))
train_data['q1_processed'] = train_data['question1'].apply(lambda x: re.sub(r"\'s", " ",str(x)))
train_data['q1_processed'] = train_data['question1'].apply(lambda x: re.sub(r"\'ve", " have ",str(x)))
train_data['q1_processed'] = train_data['question1'].apply(lambda x: re.sub(r"can't", "cannot ",str(x)))
train_data['q1_processed'] = train_data['question1'].apply(lambda x: re.sub(r"n't", " not ",str(x)))
train_data['q1_processed'] = train_data['question1'].apply(lambda x: re.sub(r"i'm", "i am ",str(x)))
train_data['q1_processed'] = train_data['question1'].apply(lambda x: re.sub(r"\'re", " are ",str(x)))
train_data['q1_processed'] = train_data['question1'].apply(lambda x: re.sub(r"\'d", " would ",str(x)))
train_data['q1_processed'] = train_data['question1'].apply(lambda x: re.sub(r"\'ll", " will ",str(x)))

train_data['q2_processed'] = train_data['question2'].apply(lambda x: re.sub(r"what's", "what is ",str(x)))
train_data['q2_processed'] = train_data['question2'].apply(lambda x: re.sub(r"\'s", " ",str(x)))
train_data['q2_processed'] = train_data['question2'].apply(lambda x: re.sub(r"\'ve", " have ",str(x)))
train_data['q2_processed'] = train_data['question2'].apply(lambda x: re.sub(r"can't", "cannot ",str(x)))
train_data['q2_processed'] = train_data['question2'].apply(lambda x: re.sub(r"n't", " not ",str(x)))
train_data['q2_processed'] = train_data['question2'].apply(lambda x: re.sub(r"i'm", "i am ",str(x)))
train_data['q2_processed'] = train_data['question2'].apply(lambda x: re.sub(r"\'re", " are ",str(x)))
train_data['q2_processed'] = train_data['question2'].apply(lambda x: re.sub(r"\'d", " would ",str(x)))
train_data['q2_processed'] = train_data['question2'].apply(lambda x: re.sub(r"\'ll", " will ",str(x)))

test_data['q1_processed'] = test_data['question1'].apply(lambda x: re.sub(r"what's", "what is ",str(x)))
test_data['q1_processed'] = test_data['question1'].apply(lambda x: re.sub(r"\'s", " ",str(x)))
test_data['q1_processed'] = test_data['question1'].apply(lambda x: re.sub(r"\'ve", " have ",str(x)))
test_data['q1_processed'] = test_data['question1'].apply(lambda x: re.sub(r"can't", "cannot ",str(x)))
test_data['q1_processed'] = test_data['question1'].apply(lambda x: re.sub(r"n't", " not ",str(x)))
test_data['q1_processed'] = test_data['question1'].apply(lambda x: re.sub(r"i'm", "i am ",str(x)))
test_data['q1_processed'] = test_data['question1'].apply(lambda x: re.sub(r"\'re", " are ",str(x)))
test_data['q1_processed'] = test_data['question1'].apply(lambda x: re.sub(r"\'d", " would ",str(x)))
test_data['q1_processed'] = test_data['question1'].apply(lambda x: re.sub(r"\'ll", " will ",str(x)))

test_data['q2_processed'] = test_data['question2'].apply(lambda x: re.sub(r"what's", "what is ",str(x)))
test_data['q2_processed'] = test_data['question2'].apply(lambda x: re.sub(r"\'s", " ",str(x)))
test_data['q2_processed'] = test_data['question2'].apply(lambda x: re.sub(r"\'ve", " have ",str(x)))
test_data['q2_processed'] = test_data['question2'].apply(lambda x: re.sub(r"can't", "cannot ",str(x)))
test_data['q2_processed'] = test_data['question2'].apply(lambda x: re.sub(r"n't", " not ",str(x)))
test_data['q2_processed'] = test_data['question2'].apply(lambda x: re.sub(r"i'm", "i am ",str(x)))
test_data['q2_processed'] = test_data['question2'].apply(lambda x: re.sub(r"\'re", " are ",str(x)))
test_data['q2_processed'] = test_data['question2'].apply(lambda x: re.sub(r"\'d", " would ",str(x)))
test_data['q2_processed'] = test_data['question2'].apply(lambda x: re.sub(r"\'ll", " will ",str(x)))


# In[4]:


# Remove all non-alphabet and non-numeric characters
train_data['q1_processed'] = train_data['q1_processed'].apply(lambda x: re.sub(r'[^\w+]'," ",str(x)))
train_data['q2_processed'] = train_data['q2_processed'].apply(lambda x: re.sub(r'[^\w+]'," ",str(x)))

test_data['q1_processed'] = test_data['q1_processed'].apply(lambda x: re.sub(r'[^\w+]'," ",str(x)))
test_data['q2_processed'] = test_data['q2_processed'].apply(lambda x: re.sub(r'[^\w+]'," ",str(x)))


# In[5]:


# Convert to lowercase and split into words
train_data['q1_processed'] = train_data['q1_processed'].apply(lambda x: str(x).lower().split())
train_data['q2_processed'] = train_data['q2_processed'].apply(lambda x: str(x).lower().split())

test_data['q1_processed'] = test_data['q1_processed'].apply(lambda x: str(x).lower().split())
test_data['q2_processed'] = test_data['q2_processed'].apply(lambda x: str(x).lower().split())


# In[6]:


# Remove stopwords
stopwords_en = set(stopwords.words('english'))
train_data['q1_processed'] = train_data['q1_processed'].apply(lambda x: [w for w in x if w not in stopwords_en])
train_data['q2_processed'] = train_data['q2_processed'].apply(lambda x: [w for w in x if w not in stopwords_en])

test_data['q1_processed'] = test_data['q1_processed'].apply(lambda x: [w for w in x if w not in stopwords_en])
test_data['q2_processed'] = test_data['q2_processed'].apply(lambda x: [w for w in x if w not in stopwords_en])


# In[7]:


print("Max length of processed q1: %d" % max(train_data['q1_processed'].apply(len)))
print("Max length of processed q2: %d" % max(train_data['q2_processed'].apply(len)))
print("99 percentile length of processed q1: %d" % np.percentile(train_data['q1_processed'].apply(len),99))
print("99 percentile length of processed q1: %d" % np.percentile(train_data['q1_processed'].apply(len),99))


# In[8]:


# Join words to form a single string
train_data['q1_processed'] = train_data['q1_processed'].apply(lambda x: " ".join(x))
train_data['q2_processed'] = train_data['q2_processed'].apply(lambda x: " ".join(x))

test_data['q1_processed'] = test_data['q1_processed'].apply(lambda x: " ".join(x))
test_data['q2_processed'] = test_data['q2_processed'].apply(lambda x: " ".join(x))


# ### Tokenize

# In[9]:


# Collate questions in a list
q1_list = train_data['q1_processed'].tolist()
q2_list = train_data['q2_processed'].tolist()

q1_list_test = test_data['q1_processed'].tolist()
q2_list_test = test_data['q2_processed'].tolist()

all_q_list = q1_list + q2_list + q1_list_test + q2_list_test
is_duplicate = train_data['is_duplicate'].tolist()


# In[10]:


tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(all_q_list)
word_index = tokenizer.word_index
print("Length of word_index: %d" % len(word_index))


# In[11]:


q1_word_seq = tokenizer.texts_to_sequences(q1_list)
q2_word_seq = tokenizer.texts_to_sequences(q2_list)
q1_word_seq_padded = pad_sequences(q1_word_seq, maxlen=MAX_SEQUENCE_LENGTH)
q2_word_seq_padded = pad_sequences(q2_word_seq, maxlen=MAX_SEQUENCE_LENGTH)
y_train = np.array(is_duplicate, dtype=int)

q1_word_seq_test = tokenizer.texts_to_sequences(q1_list_test)
q2_word_seq_test = tokenizer.texts_to_sequences(q2_list_test)
q1_word_seq_padded_test = pad_sequences(q1_word_seq_test, maxlen=MAX_SEQUENCE_LENGTH)
q2_word_seq_padded_test = pad_sequences(q2_word_seq_test, maxlen=MAX_SEQUENCE_LENGTH)


# ### GloVe embedding

# In[12]:


# Obtain GloVe embeddings file from "http://nlp.stanford.edu/data/glove.840B.300d.zip"
# Create a dictionary with word:vector as key:value pair
embed_dict = {}
with open("glove.840B.300d.txt", encoding='utf-8') as f:
    for line in f:
        temp_list = line.split(' ')
        word = temp_list[0]
        embed_vec = np.asarray(temp_list[1:], dtype='float32')
        embed_dict[word] = embed_vec

print('Word embeddings: %d' % len(embed_dict))


# ### Embedding Matrix

# In[13]:


nb_words = min(MAX_NB_WORDS, len(word_index))
embed_matrix = np.zeros((nb_words + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    if word in embed_dict:
        embed_matrix[i] = embed_dict[word]

print(embed_matrix.shape)
print('Null word embeddings: %d' % np.sum(np.sum(embed_matrix, axis=1) == 0))


# In[14]:


import pickle
pickle.dump([q1_list,q2_list,q1_list_test,q2_list_test], open("q_list.p", "wb"))
pickle.dump(embed_dict, open("embed_dict.p", "wb"))


# ## Neural Network Architecture

# In[15]:


from keras.models import Model
from keras.layers import Input, Dense, Dropout, Lambda, Reshape, Flatten, TimeDistributed, BatchNormalization
from keras.layers import concatenate, GlobalAveragePooling2D, Conv2D, MaxPooling2D, dot
from keras.layers.embeddings import Embedding
from keras import backend as K
from keras.callbacks import Callback, ModelCheckpoint
from sklearn.model_selection import train_test_split


# In[16]:


def sq_diff(pair_of_tensors):
    x, y = pair_of_tensors
    return K.square(x - y)

def abs_diff(pair_of_tensors):
    x, y = pair_of_tensors
    return K.abs(x - y)


# ### Train model

# In[23]:


question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

q1 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[embed_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question1)



q2 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[embed_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question2)

square_diff = Lambda(sq_diff)([q1, q2])
square_diff = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(square_diff)

square_diff = Conv2D(filters=64, kernel_size=5, strides=1, activation='relu')(square_diff)
square_diff = MaxPooling2D(pool_size=2)(square_diff)
square_diff = Conv2D(filters=64, kernel_size=4, strides=1, activation='relu')(square_diff)
square_diff = MaxPooling2D(pool_size=2)(square_diff)
square_diff = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(square_diff)
square_diff = GlobalAveragePooling2D()(square_diff)

absolute_diff = Lambda(abs_diff)([q1, q2])
absolute_diff = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(absolute_diff)

absolute_diff = Conv2D(filters=64, kernel_size=5, strides=1, activation='relu')(absolute_diff)
absolute_diff = MaxPooling2D(pool_size=2)(absolute_diff)
absolute_diff = Conv2D(filters=64, kernel_size=4, strides=1, activation='relu')(absolute_diff)
absolute_diff = MaxPooling2D(pool_size=2)(absolute_diff)
absolute_diff = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(absolute_diff)
absolute_diff = GlobalAveragePooling2D()(absolute_diff)

q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
q1_max = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)
q2_max = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)

ensemble = concatenate([square_diff,absolute_diff,q1_max,q2_max])

ensemble = Dense(256, activation='relu')(ensemble)
ensemble = Dropout(0.1)(ensemble)
ensemble = BatchNormalization()(ensemble)
ensemble = Dense(256, activation='relu')(ensemble)
ensemble = Dropout(0.1)(ensemble)
ensemble = BatchNormalization()(ensemble)
ensemble = Dense(256, activation='relu')(ensemble)
ensemble = Dropout(0.1)(ensemble)
ensemble = BatchNormalization()(ensemble)
ensemble = Dense(256, activation='relu')(ensemble)
ensemble = Dropout(0.1)(ensemble)
ensemble = BatchNormalization()(ensemble)

is_duplicate = Dense(1, activation='sigmoid')(ensemble)

model = Model(inputs=[question1,question2], outputs=is_duplicate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# In[ ]:


checkpointer = ModelCheckpoint(filepath='quora_q_pairs.cnn_model_ensemble.best.hdf5',verbose=1, save_best_only=True)

results = model.fit([q1_word_seq_padded,q2_word_seq_padded], y_train, batch_size=128, epochs=10,
          validation_split=0.2, callbacks=[checkpointer],
          verbose=2, shuffle=True)


# #### Add cosine similarity feature

# In[20]:


question1 = Input(shape=(MAX_SEQUENCE_LENGTH,))
question2 = Input(shape=(MAX_SEQUENCE_LENGTH,))

q1 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[embed_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question1)



q2 = Embedding(nb_words + 1, 
                 EMBEDDING_DIM, 
                 weights=[embed_matrix], 
                 input_length=MAX_SEQUENCE_LENGTH, 
                 trainable=False)(question2)

square_diff = Lambda(sq_diff)([q1, q2])
square_diff = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(square_diff)

square_diff = Conv2D(filters=64, kernel_size=5, strides=1, activation='relu')(square_diff)
square_diff = MaxPooling2D(pool_size=2)(square_diff)
square_diff = Conv2D(filters=64, kernel_size=4, strides=1, activation='relu')(square_diff)
square_diff = MaxPooling2D(pool_size=2)(square_diff)
square_diff = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(square_diff)
square_diff = GlobalAveragePooling2D()(square_diff)

absolute_diff = Lambda(abs_diff)([q1, q2])
absolute_diff = Reshape((MAX_SEQUENCE_LENGTH, EMBEDDING_DIM, 1))(absolute_diff)

absolute_diff = Conv2D(filters=64, kernel_size=5, strides=1, activation='relu')(absolute_diff)
absolute_diff = MaxPooling2D(pool_size=2)(absolute_diff)
absolute_diff = Conv2D(filters=64, kernel_size=4, strides=1, activation='relu')(absolute_diff)
absolute_diff = MaxPooling2D(pool_size=2)(absolute_diff)
absolute_diff = Conv2D(filters=64, kernel_size=3, strides=1, activation='relu')(absolute_diff)
absolute_diff = GlobalAveragePooling2D()(absolute_diff)

q1 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q1)
q2 = TimeDistributed(Dense(EMBEDDING_DIM, activation='relu'))(q2)
q1_max = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q1)
q2_max = Lambda(lambda x: K.max(x, axis=1), output_shape=(EMBEDDING_DIM, ))(q2)

q_dot = dot([q1_max,q2_max], [1,1], normalize=True)

ensemble = concatenate([square_diff,absolute_diff,q1_max,q2_max,q_dot])

ensemble = Dense(256, activation='relu')(ensemble)
ensemble = Dropout(0.1)(ensemble)
ensemble = BatchNormalization()(ensemble)
ensemble = Dense(256, activation='relu')(ensemble)
ensemble = Dropout(0.1)(ensemble)
ensemble = BatchNormalization()(ensemble)
ensemble = Dense(256, activation='relu')(ensemble)
ensemble = Dropout(0.1)(ensemble)
ensemble = BatchNormalization()(ensemble)
ensemble = Dense(256, activation='relu')(ensemble)
ensemble = Dropout(0.1)(ensemble)
ensemble = BatchNormalization()(ensemble)

is_duplicate = Dense(1, activation='sigmoid')(ensemble)

model = Model(inputs=[question1,question2], outputs=is_duplicate)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()


# In[21]:


checkpointer = ModelCheckpoint(filepath='quora_q_pairs.cnn_model_ensembleII.best.hdf5',verbose=1, save_best_only=True)

results = model.fit([q1_word_seq_padded,q2_word_seq_padded], y_train, batch_size=128, epochs=10,
          validation_split=0.2, callbacks=[checkpointer],
          verbose=2, shuffle=True)


# ### Predict Probabilities of test data

# In[24]:


model.load_weights('quora_q_pairs.cnn_model_ensemble.best.hdf5')
p = model.predict([q1_word_seq_padded_test,q2_word_seq_padded_test], batch_size=128, verbose=0)


# In[40]:


output = pd.DataFrame(p)
output.columns = ['is_duplicate']
output['test_id'] = output.index
output = output[['test_id','is_duplicate']]
output.head()


# In[43]:


output.to_csv("prediction.csv",index=False)

