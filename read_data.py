import pickle 
from cv2 import imread
a = pickle.load(open('train_imdb1.pickle','rb'))
for key in a.keys():
	for i in range(len(a[key]['videos'])):
		for j in range(len(a[key]['videos'][i])):
			a[key]['videos'][i][j] = imread(a[key]['videos'][i][j])
pickle.dump(a,open('data.pickle','wb'))				