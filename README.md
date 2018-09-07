# Association-Net
This is an unofficial implementation of the Paper  "Learning Feature Hierarchies from Long-Range Temporal Associations in Videos"  By Panna Felsen Katerina Fragkiadaki Jitendra Malik Alexei Efros

Data Requirement
We would be training our AssociationNet using object tubes dataset

Get Dataset from the Image Net website
Augment the folder ILSVRC2017 with the new data

Run scripts/preprocess_VID_Data.py

As we are working with object tubes running it  creates two types of images 
i- centered around the object in video   
ii - object not centered around the object (In this case we shift the center by a distance)

Next run scripts/build_VID2015_imdb.py

Creates pickled file for the dataset

The credit for the base script goes to ''. Although I have made few changes in it to fit our data requirements.

Run python train.py

The trained model is saved in model.h5 file

