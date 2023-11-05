# Importing the libraries
import joblib
from transformers import AutoTokenizer, TFAutoModel
from transformers import TFAutoModelForSequenceClassification
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense
import numpy as np
print("Import Successfull")

os.system("clear")

# importing the dataset 
df_train = pd.read_csv('./train.csv',  encoding='utf-8')
df_test = pd.read_csv('./test.csv', encoding='utf-8')
#print(df_test.head())

from tensorflow.keras.utils import to_categorical

y_train = to_categorical(df_train.label)
y_test = to_categorical(df_test.label)

os.system("clear")


print("\ny_train shape,",y_train.shape)
print("\ny_test shape,",y_test.shape)

#tokenizer = AutoTokenizer.from_pretrained('dmis-lab/biobert-v1.1')
#model = TFAutoModelForSequenceClassification.from_pretrained("dmis-lab/biobert-v1.1", from_pt=True)

from transformers import AutoTokenizer,TFBertModel
tokenizer = AutoTokenizer.from_pretrained('./save_tokenizer/')
bert_model = TFBertModel.from_pretrained("./save_model/", from_pt=True)


# Tokenize the input (takes some time) 
# here tokenizer using from bert-base-cased
x_train = tokenizer(
    text=df_train.text.tolist(),
    add_special_tokens=True,
    max_length=100,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)
x_test = tokenizer(
    text=df_test.text.tolist(),
    add_special_tokens=True,
    max_length=100,
    truncation=True,
    padding=True, 
    return_tensors='tf',
    return_token_type_ids = False,
    return_attention_mask = True,
    verbose = True)

input_ids = x_train['input_ids']
attention_mask = x_train['attention_mask']

print(x_test['input_ids'].shape)
os.system("clear")


max_len =38 
input_ids = Input(shape=(max_len,), dtype=tf.int32, name="input_ids")
input_mask = Input(shape=(max_len,), dtype=tf.int32, name="attention_mask")
embeddings = bert_model(input_ids,attention_mask = input_mask)[0] 
print('Embeddings shape: {}'.format(embeddings.shape))
#embeddings = bert_model.bertmodel(input_ids, input_mask)[0]
out = tf.keras.layers.GlobalMaxPool1D()(embeddings)
out = Dense(128, activation='relu')(out)
out = tf.keras.layers.Dropout(0.1)(out)
out = Dense(32,activation = 'relu')(out)
y = Dense(2,activation = 'sigmoid')(out)
model = tf.keras.Model(inputs=[input_ids, input_mask], outputs=y)
model.layers[2].trainable = True

print("\n\nModel Compilation\n\n")

optimizer = Adam(
    learning_rate=5e-05, # this learning rate is for bert model , taken from huggingface website
    epsilon=1e-08,
    decay=0.01,
    clipnorm=1.0)


# Set loss and metrics
loss =CategoricalCrossentropy(from_logits = True)
metric = CategoricalAccuracy('balanced_accuracy'),
# Compile the model


model.compile(
    optimizer = optimizer,
    loss = loss,
    metrics = metric)


print("Model training")


train_history = model.fit(
    x ={'input_ids':x_train['input_ids'],'attention_mask':x_train['attention_mask']} ,
    y = y_train,
    validation_data = (
    {'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']}, y_test
    ),
  epochs=15,
    batch_size=36
)

#tf.saved_model.save(model,'./save_model_tf/')
model.save('./save_model_tf/')

#joblib.dump(model, './save_model_tf/model.pkl')


from tensorflow import keras

predicted_raw = model.predict({'input_ids':x_test['input_ids'],'attention_mask':x_test['attention_mask']})


y_predicted = np.argmax(predicted_raw, axis = 1)
y_true = df_test.label

print("Predicted",y_predicted)
print("Actual",str(y_true))


from sklearn.metrics import classification_report
print(classification_report(y_true, y_predicted))

import nltk

for i in range(30):
    texts = input(str('input the text'))
    sent_text = nltk.sent_tokenize(texts)
    for i in sent_text:
        x_val = tokenizer(
            text=i,
            add_special_tokens=True,
            max_length=38,
            truncation=True,
            padding='max_length', 
            return_tensors='tf',
            return_token_type_ids = False,
            return_attention_mask = True,
            verbose = True) 
        validation = model.predict({'input_ids':x_val['input_ids'],'attention_mask':x_val['attention_mask']})*100
        encoded_dict = {'functional_deficits':0,'delay_reason':1}
        for key , value in zip(encoded_dict.keys(),validation[0]):
            print(key,value)
