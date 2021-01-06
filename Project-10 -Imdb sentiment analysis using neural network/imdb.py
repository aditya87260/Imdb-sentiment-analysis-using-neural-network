from keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

((XT,YT),(Xt,Yt))=imdb.load_data(num_words=10000)

print(len(XT),len(Xt))

print(XT[0])
word_idx=imdb.get_word_index()
print(word_idx.items())

idx_word=dict([value,key] for (key,value) in word_idx.items())

print(idx_word)

actual_review=" ".join([idx_word.get(idx-3,"?") for idx in XT[0]])

print(actual_review)

#Vectorization of Data
#Transform the Data into vector of fixed length.

#Vocab size-10,000 we will make sure every sentence is represented by a vector of len 10,000 i.e,[010100001110010.....10010]

#we can achieve this creating a function which can also be done by count vectorization method

def vectorize_sentences(sentences,dim=10000):
  outputs=np.zeros((len(sentences),dim))

  for i,idx in enumerate(sentences):
    outputs[i,idx]=1

  return outputs

X_train = vectorize_sentences(XT)
X_test = vectorize_sentences(Xt)

print(X_train.shape)
print(X_test.shape)

Y_train = np.asarray(YT).astype("float32")
Y_test = np.asarray(Yt).astype("float32")

print(YT)
print(Y_train)

#Model Architecture
#Use fully Connected/Dense Layers with Relu as Activation function
#2 Hidden Layers with 16 Units each
#1 Output Layer with 1 unit which uses Sigmoid as an Activation fucntion.

from keras import models
from keras.layers import Dense

model=models.Sequential()
model.add(Dense(16,activation="relu",input_shape=(10000,)))
model.add(Dense(16,activation="relu"))
model.add(Dense(1,activation="sigmoid"))

# compile the model
model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
model.summary()

#Training And Validation
X_val=X_train[:5000]
X_train_new=X_train[5000:]

Y_val=Y_train[:5000]
Y_train_new=Y_train[5000:]

print(X_train_new.shape)

hist=model.fit(X_train_new,Y_train_new,epochs=4,batch_size=512,validation_data=(X_val,Y_val))

print(hist)

#Visualise our Results

h=hist.history
plt.plot(h["val_loss"],label="Validation Loss")
plt.plot(h["loss"],label="Training Loss")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

plt.plot(h["val_accuracy"],label="Validation Accuracy")
plt.plot(h["accuracy"],label="Training Accuracy")
plt.legend()
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.show()

#Checking Accuracy

print(model.evaluate(X_test,Y_test)[1])
print(model.evaluate(X_train_new,Y_train_new)[1])

#Predictions

outputs=[]
outputs=model.predict(X_test)
print(outputs)

for  i in range(25000):
  if outputs[i] >0.5:
       outputs[i]=1
  else:
       outputs[i]=0
print(outputs)