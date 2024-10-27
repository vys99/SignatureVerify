#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import pickle
import tkinter as tk
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tkinter import filedialog
from PIL import Image,ImageTk
import pyttsx3
from sklearn.metrics import roc_curve, auc


# In[8]:


with open('Signature model_05_95.pkl','rb') as f:
    model = pickle.load(f)


# In[9]:


engine = pyttsx3.init()


# In[10]:


def upload_file():
    global file_path
    file_path = filedialog.askopenfilename(
        title="Select a File", 
        filetypes=[("All Files", "."), ("Text Files", ".txt"), ("Image Files", ".png;.jpg;.jpeg")]
    )
    if file_path:
        print(f"Selected file: {file_path}")
        image = Image.open(file_path)
        image = image.resize((150,100), Image.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        label.config(image=photo)
        label.photo = photo
        label1.config(text='Uploaded',bg='ivory')
        
def verify():
    img_size=100
    img = cv2.imread(file_path)
    grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(grey,(img_size,img_size))
    image = resized / 255.0
    inp_dat = image.reshape((1,100,100,1))
    pred = model.predict(inp_dat)
    print(pred[0][0])
    if pred > 0.5:
        label2.config(text='GENUINE',bg='ivory',fg='green',font=("Helvetica", 22,'bold'))
        text='The signature is Genuine'
        engine.say(text)
        engine.runAndWait()
        
    else:
        label2.config(text='FAKE',bg='ivory',fg='red',font=("Helvetica", 22,'bold'))
        text='The Signature is Fake'
        engine.say(text)
        engine.runAndWait()

def clear_all():
    label.config(image='',bg='#fe4e3d')
    label1.config(text='',bg='#fe4e3d')
    label2.config(text='',bg='#fe4e3d')


# In[11]:


root = tk.Tk()
root.title('Signature Verification')
root.config(bg='#ff9696')
root.geometry('700x650')

frame = tk.Frame(root, bg='#fe4e3d', width=450, height=400, relief=tk.RAISED, borderwidth=5)
frame.pack(padx=30, pady=30)
frame.pack(fill=tk.BOTH, expand=True)


label = tk.Label(frame,text = 'SIGNATURE VERIFICATION',fg='ivory',bg='#fe4e3d',font=("Verdana", 28,'bold'))
label.pack(pady=20)

upload_button = tk.Button(frame, text="Upload Signature",bg = '#ffd100', command=upload_file,font=("Helvetica", 16),width=20,height = 2)
upload_button.pack(pady=20)

label = tk.Label(frame, image='',bg='#fe4e3d')
label.pack(pady=10)

label1 = tk.Label(frame,text = '',font=("Helvetica", 18),bg='#fe4e3d')
label1.pack(pady=10)

button2 = tk.Button(frame,text='Verify Signature',bg = '#ffd100',command = verify,font=("Helvetica", 16),width=20,height = 2,)
button2.pack(pady = 10)

label2 = tk.Label(frame,text = '',font=("Helvetica", 18),bg='#fe4e3d')
label2.pack(pady=20)

clean_button = tk.Button(frame,text='Clear All',command=clear_all,font=("Helvetica", 16),width=20,height = 2,bg='#ffd100', fg="black")
clean_button.pack(pady=20)

root.mainloop()


# In[ ]:




