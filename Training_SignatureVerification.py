#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,MaxPooling2D,Flatten,Dropout,BatchNormalization,Input
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_curve, auc
import seaborn as sns
from sklearn.metrics import confusion_matrix


# In[13]:


categories = ['Genuine','Fake']


# In[14]:


categories


# In[15]:


img_size = 100
x = []
y = []
for category in categories:
    folder_path = os.path.join(r"C:\Users\vysak\OneDrive\Desktop\archive (14) - Copy",category)
    img_folder_names = os.listdir(folder_path)
    for img_folder_name in img_folder_names:
        img_folder_path = os.path.join(folder_path,img_folder_name)
        image_names = os.listdir(img_folder_path)
        for img_name in image_names:
            path = os.path.join(img_folder_path,img_name)
            img = cv2.imread(path)
            grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            resized = cv2.resize(grey,(img_size,img_size))
            x.append(resized)
            y.append(category)


# In[16]:


x = np.array(x)
x


# In[17]:


print(len(x))


# In[18]:


le =LabelEncoder()
y = le.fit_transform(y)
y


# In[19]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[20]:


fig =plt.figure(figsize = (15,15))
for i in range(100):
    ax = fig.add_subplot(10,10,i+1,xticks = [],yticks = [])
    ax.imshow(np.squeeze(x_train[i]),cmap = 'gray')
    ax.set_title(categories[y_train[i]])
plt.subplots_adjust(hspace=0.4)


# In[21]:


x_train.shape


# In[22]:


x_test.shape


# In[23]:


x_train = x_train.astype('float32')/255.0
x_test = x_test.astype('float32')/255.0


# In[25]:


x_train=x_train.reshape((1319,100,100,1))
x_test = x_test.reshape((330, 100,100,1))


# In[46]:


model = Sequential()
Input(shape=(100, 100, 1)),
model.add(Conv2D(32, (3, 3), activation='relu')),
# model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 1)))
model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D((2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))


# In[47]:


model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


# In[48]:


from tensorflow.keras.callbacks import EarlyStopping

early_stopping = EarlyStopping(monitor='val_loss', patience=3)
history = model.fit(x_train, y_train, epochs=10, batch_size=10, validation_data=(x_test, y_test), callbacks=[early_stopping])


# In[3]:


model.summary()


# In[2]:


with open('Signature model_05_95.pkl','rb') as f:
    model = pickle.load(f)


# In[49]:


loss, accuracy = model.evaluate(x_test, y_test, verbose=0)
print('Loss :',loss*100,'%')
print('Accuracy :',accuracy*100,'%')


# In[29]:


with open('Signature model_04.pkl','wb') as f:
    pickle.dump(model,f)


# In[50]:


with open('Signature model_05_95.pkl','wb') as f:
    pickle.dump(model,f)


# In[59]:


with open('Signature model_02.pkl','wb') as f:
    pickle.dump(model,f)


# In[51]:


y_pred_proba = model.predict(x_test)


# In[52]:


y_pred_proba


# In[55]:


fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)


# In[56]:


roc_auc = auc(fpr, tpr)


# In[57]:


plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # Diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[62]:


y_pred_proba = model.predict(x_test)
y_pred = (y_pred_proba > 0.5).astype(int)


# In[64]:


cm = confusion_matrix(y_test, y_pred)


# In[65]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted Negative', 'Predicted Positive'], 
            yticklabels=['Actual Negative', 'Actual Positive'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




