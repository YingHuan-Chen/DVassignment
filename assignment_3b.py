#!/usr/bin/env python
# coding: utf-8

# In[34]:


import os
import cv2
import numpy as np
import pickle

# enter your code here
def get_data_from_file(data_dir,file):
    data=[]
    with open(file,'r') as fp:
        for line in fp:
            #print(line)
            img_path=os.path.join(data_dir,line)
            print(img_path)
            img =cv2.imread(img_path)
            #img_resized =cv2.resize(img,(48,48))
            if not(line.find('Cat')==-1):
                label =0
            elif not(line.find('Dog')==-1):
                label =1
            #print(label)
            data.append([img,label])
           
                
    return data

def get_image_and_label(data):
    X =data[:,0]
    Y =data[:,1]
    return X,Y

if __name__ == '__main__':
    data_dir = '/Desktop/DVassignment/PetImages'
    train_file = 'train_list.txt'
    test_file = 'test_list.txt'

    # enter your code here
    train_data =get_data_from_file(data_dir,train_file)
    print(train_data)
    X_train,Y_train =get_image_and_label(train_data)
    
    test_data =get_data_from_file(data_dir,test_file)
    X_test,Y_test =get_image_and_label(test_data)
    
    print(len(train_data))
    print(train_data[0][0].shape)
    print(len(test_data))
    print(test_data[0][0].shape)
    print(X_train.shape)
    print(Y_train.shape)

    save_path = 'dogs_cats.pkl'
    print('Saving to', save_path)
    data = {}
    data['X_train'] = X_train
    data['Y_train'] = Y_train
    data['X_test'] = X_test
    data['Y_test'] = Y_test
    pickle.dump(data, open(save_path, 'wb'))


# In[ ]:




