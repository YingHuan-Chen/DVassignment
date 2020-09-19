import os
import cv2
import numpy as np
import pickle

def get_data_from_file(file):
	data=[]
	with open(file) as fp:
		for line in fp:
			img_path=os.path.join('PetImages',line[:-1])
			try:
				img =cv2.imread(img_path)
				img_resized =cv2.resize(img,(48,48))
				if not(line.find('Cat')==-1):
 					label =0
				elif not(line.find('Dog')==-1):
					label =1
				data.append([img_resized,label])
			except:
				print('error')
	return data
	
def get_image_and_label(data):
	X=[]
	Y=[]
	for i,j in data:
		X.append(i)
		Y.append(j)
	return X,Y

if __name__ == '__main__':
	train_file='train_list.txt'
	test_file='test_list.txt'
	train_data =get_data_from_file(train_file)
	X_train,Y_train =get_image_and_label(train_data)
	test_data =get_data_from_file(test_file)
	X_test,Y_test =get_image_and_label(test_data)
	X_train =np.array(X_train)
	Y_train =np.array(Y_train)
	X_test =np.array(X_test)
	Y_test =np.array(Y_test)
	#print(train_data)
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
