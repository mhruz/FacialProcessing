# Facial Processing

You need to create a directory in the root called "models".

## detect_and_save.py 
Detects faces and facial landmarks in images using dlib (http://dlib.net/). Saves them as JSON file.
You need to download the model for landmarks (http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
and save the unpacked model file into the directory "models".

The locations are in relative coordinates. They follow the DLIB structure which uses iBUG facial keypoint structure
(https://ibug.doc.ic.ac.uk/resources/facial-point-annotations/).

Requirements:  
dlib==19.21.0  
numpy==1.19.1  
six==1.15.0  

