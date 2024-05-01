import os 
import cv2
import numpy as np 
import random
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import img_to_array, load_img
#Function to extract frames from a video


def extract_frames(video_path, num_frames=8, resize_shape=(112, 112)):
    vidcap = cv2.VideoCapture(video_path)
    total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = total_frames / (num_frames - 1)
    frames_to_capture = [int(interval * i) for i in range(num_frames)]
    frames = []
    count = 0
    success = True
    while success:
        success, image = vidcap.read()
        if count in frames_to_capture:
            if success:
                image = cv2.resize(image, resize_shape)
                image = image.astype(np.float32) / 255.0
                frames.append(image)
        count += 1
    vidcap.release()
    return frames

# Function to preprocess image
def preprocess_image(image, target_size=(112,112)):
    image = cv2.resize(image, target_size)
    image = img_to_array(image)
    return image


def main():
    # Path to fake and real videos folders
    fake_videos_folder = "data/fake"
    real_videos_folder = "data/real"


    # Extract frames from fake and real video folders
    fake_X = []
    fake_y = []
    for video_file in os.listdir(fake_videos_folder):
        video_path = os.path.join(fake_videos_folder, video_file)
        frames = extract_frames(video_path)
        fake_X.extend(frames)
        fake_y.extend([0] * len(frames))

    print("fake")

    real_X = []
    real_y = []
    for video_file in os.listdir(real_videos_folder):
        video_path = os.path.join(real_videos_folder, video_file)
        frames = extract_frames(video_path)
        real_X.extend(frames)
        real_y.extend([1] * len(frames))

    print("real")

    # Combine fake and real data
    X = np.array(fake_X + real_X)
    y = np.array(fake_y + real_y)


    # Shuffle data
    random.seed(42)
    random.shuffle(X)
    random.seed(42)
    random.shuffle(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


    # Preprocess images
    X_train = np.array([preprocess_image(image) for image in X_train])
    X_test = np.array([preprocess_image(image) for image in X_test])

    print("Done with preprocessing.")

    # One-hot encode labels
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)

    y_train_one_hot = to_categorical(y_train_encoded)
    y_test_one_hot = to_categorical(y_test_encoded)

    # Compare original labels with encoded labels
    print("Original labels:", y_train[:5])  # Display first 5 original labels
    print("Encoded labels:", y_train_encoded[:5])  # Display first 5 encoded labels
    print("One-hot encoded labels:", y_train_one_hot[:5])

    # dividing and test and validation  

    X_test, X_val, y_test_one_hot, y_val = train_test_split(X_test, y_test_one_hot, test_size=0.5, random_state=42)



    print(f"""
          Training data shape (X,y) : ({X_train.shape,y_train_one_hot.shape})
          Testing data shape (X,y)  : ({X_test.shape,y_test_one_hot.shape})
          Validation data shape (X,y) :({X_val,y_val})
    """)

    np.save('X_train.npy',X_train)
    np.save('y_train.npy',y_train_one_hot)
    np.save('y_test.npy',y_test_one_hot)
    np.save('X_test.npy',X_test)
    np.save('X_val.npy',X_val)
    np.save('y_val.npy',y_val)



    # os.system('split -b 50 X_train.npy X_train_split_ --additional-suffix .part')
    # os.system(' split -b 50 X_test.npy X_test_split_ --additional-suffix .part')
    # os.system('  split -b 50 X_val.npy X_val_split_ --additional-suffix .part')
    # os.system('mv X_* y_* ../data')
    # combining them 
    # os. system('for i in data/X_train*:do cat $i >> X_train.npy')
    # os. system('for i in data/X_test*:do cat $i >> X_test.npy')
    # os. system('for i in data/X_val*:do cat $i >> X_val.npy')

main()