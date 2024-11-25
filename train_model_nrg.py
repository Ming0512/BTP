# import the necessary packages
from imutils import paths
import face_recognition
import pickle
import cv2
import os

dataset_path = "dataset"

print("[INFO] Start processing faces...")
imagePaths = list(paths.list_images(dataset_path))

knownEncodings = []
knownNames = []
knownRollNumbers = []

for (i, imagePath) in enumerate(imagePaths):
    print(f"[INFO] Processing image {i + 1}/{len(imagePaths)}")
    folder_name = imagePath.split(os.path.sep)[-2]
    name, roll_number = folder_name.split('_')
    image = cv2.imread(imagePath)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")  
    encodings = face_recognition.face_encodings(rgb, boxes)
    for encoding in encodings:
        knownEncodings.append(encoding)
        knownNames.append(name)
        knownRollNumbers.append(roll_number)

print("[INFO] Serializing encodings to disk...")
data = {
    "encodings": knownEncodings,
    "names": knownNames,
    "roll_numbers": knownRollNumbers,
}
with open("encodings.pickle", "wb") as f:
    f.write(pickle.dumps(data))

print("[INFO] Face encoding completed successfully.")
