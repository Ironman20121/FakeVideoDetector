## imports

import tensorflow as tf
import numpy as np
import pandas as pd
import cv2
import os
import matplotlib.pyplot as plt


from sklearn.metrics import confusion_matrix, classification_report


###global parametes
IMG_SIZE =150
BATCH_SIZE = 64
EPOCHS = 50
MAX_SEQ_LENGTH = 20
NUM_FEATURES = 2048

trainCsvpath = "../data/train.csv"
testCsvpath = "../data/test.csv"

df = pd.read_csv("../data/train.csv")
df2 = pd.read_csv('../data/test.csv')

label_processor = tf.keras.layers.StringLookup(
    num_oov_indices=0, vocabulary=np.unique(df["tag"])
)


filepath = "../models/tmp/"
checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath, save_weights_only=True, save_best_only=True, verbose=1
    )
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='accuracy',
    patience=10,
    verbose=1,
    mode='min',
    )

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]


def load_video(path, max_frames=0, resize=(IMG_SIZE, IMG_SIZE)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]]
            frames.append(frame)

            if len(frames) == max_frames:
                break
    finally:
        cap.release()
    return np.array(frames)

def build_cnn_feature_extractor(input_shape=(IMG_SIZE, IMG_SIZE, 3)):
    # act_f ='LeakyReLU'
    # act_f = mish
    act_f='relu'
    model = tf.keras.models.Sequential([
        
        tf.keras.layers.Conv2D(32,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        
   
        
        tf.keras.layers.Conv2D(64,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),

        
        tf.keras.layers.Conv2D(64,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),

       
        tf.keras.layers.Conv2D(128,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),


        tf.keras.layers.Conv2D(256,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        

       
        tf.keras.layers.Conv2D(512,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        tf.keras.layers.MaxPooling2D((2,2)),
        
       

        tf.keras.layers.Dense(8000,activation='LeakyReLU'),
        tf.keras.layers.Dense(1024,activation='LeakyReLU'),
        tf.keras.layers.Dense(2048,activation='LeakyReLU'),
        tf.keras.layers.MaxPooling2D()
    ])
    return model

# Function to generate adversarial attacks
def adversarial_attack(classifier, x_data, y_data,model):
    # Create an instance of the DeepFool attack
    attack = classifier
    attack_name = type(classifier).__name__  # to get name of attack class 
    # Generate adversarial examples for all samples in x_data
    x_adv = attack.generate(x=x_data, y=y_data)
    test_loss_adv, test_accuracy_adv = model.evaluate(x_adv, y_data)
    print(f"Adversarial Test Loss {attack_name}:", test_loss_adv)
    print(f"Adversarial Test Accuracy {attack_name}:", test_accuracy_adv)
    return x_adv

def prepare_all_videos(df, root_dir):
        num_samples = len(df)
        video_paths = df["video_name"].values.tolist()
        labels = df["tag"].values
        labels = label_processor(labels[..., None]).numpy()

        # `frame_masks` and `frame_features` are what we will feed to our sequence model.
        # `frame_masks` will contain a bunch of booleans denoting if a timestep is
        # masked with padding or not.
        frame_masks = np.zeros(shape=(num_samples, MAX_SEQ_LENGTH), dtype="bool")
        # print(frame_masks)
        frame_features = np.zeros(
            shape=(num_samples, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
        )
        # print(frame_features)

        # For each video.
        for idx, path in enumerate(video_paths):
            # Gather all its frames and add a batch dimension.
            frames = load_video(os.path.join(root_dir, path))
            frames = frames[None, ...]

            # Initialize placeholders to store the masks and features of the current video.
            temp_frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
            temp_frame_features = np.zeros(
                shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32"
            )

            # Extract features from the frames of the current video.
            for i, batch in enumerate(frames):
                video_length = batch.shape[0]
                length = min(MAX_SEQ_LENGTH, video_length)
                for j in range(length):
                    temp_frame_features[i, j, :] = feature_extractor.predict(
                        batch[None, j, :]
                    )
                temp_frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

            frame_features[idx,] = temp_frame_features.squeeze()
            frame_masks[idx,] = temp_frame_mask.squeeze()

        return (frame_features, frame_masks), labels


def my_gru():
    class_vocab = label_processor.get_vocabulary()
    frame_features_input = tf.keras.Input((MAX_SEQ_LENGTH, NUM_FEATURES))
    mask_input = tf.keras.Input((MAX_SEQ_LENGTH,), dtype="bool")
    x = tf.keras.layers.GRU(16, return_sequences=True)(frame_features_input, mask=mask_input)
    x = tf.keras.layers.GRU(8)(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(8, activation="relu")(x)
    output = tf.keras.layers.Dense(len(class_vocab), activation="softmax")(x)

    rnn_model = tf.keras.Model([frame_features_input, mask_input], output)

    rnn_model.compile(
        loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )
    rnn_model.summary()
    return rnn_model


def prepare_single_video(frames):
    frames = frames[None, ...]
    frame_mask = np.zeros(shape=(1, MAX_SEQ_LENGTH,), dtype="bool")
    frame_features = np.zeros(shape=(1, MAX_SEQ_LENGTH, NUM_FEATURES), dtype="float32")
    feature_extractor = build_cnn_feature_extractor()
    for i, batch in enumerate(frames):
        video_length = batch.shape[0]
        length = min(MAX_SEQ_LENGTH, video_length)
        for j in range(length):
            frame_features[i, j, :] = feature_extractor.predict(batch[None, j, :])
        frame_mask[i, :length] = 1  # 1 = not masked, 0 = masked

    return frame_features, frame_mask


def sequence_prediction(path):
    class_vocab = label_processor.get_vocabulary()

    frames = load_video(os.path.join("test", path))
    rnn_model = my_gru()
    frame_features, frame_mask = prepare_single_video(frames)
    probabilities = rnn_model.predict([frame_features, frame_mask])[0]

    for i in np.argsort(probabilities)[::-1]:
        print(f"  {class_vocab[i]}: {probabilities[i] * 100:5.2f}%")
    return frames


def to_gif(images):
    converted_images = images.astype(np.uint8)
    
    imageio.mimsave("animation.gif", converted_images, fps=10)
    return embed.embed_file("animation.gif")

def trainandtest(model):
    
    X_train_features = np.load('../data/numpyarrs/X_train_features.npy')
    X_train_masks = np.load('../data/numpyarrs/X_train_masks.npy')
    y_train = np.load('../data/numpyarrs/y_train.npy')
    X_test_features = np.load('../data/numpyarrs/X_test_features.npy')
    X_test_masks = np.load('../data/numpyarrs/X_test_masks.npy')
    y_test = np.load('../data/numpyarrs/y_test.npy')

    X_train = (X_train_features,X_train_masks)
    X_test = (X_test_features,X_test_masks)

    
    
    history = model.fit(
        [X_train[0],X_train[1]],
        y_train,validation_split=0.3,
        epochs=EPOCHS,batch_size=BATCH_SIZE,
        callbacks=[checkpoint],
    )
    model.load_weights(filepath)
    model.save("../models/GRU-RNN")
    # plot_training_history(history,"trainingGrpahs.png")
    _, accuracy = model.evaluate([X_test[0], X_test[1]], y_test)
    print(f"")
    print(f"""
           Test_loss : {round(_*100,2)}%
           Test accuracy : {round(accuracy * 100, 2)}%            
""")
    print("CNNRNN Confusion Matrix and report :")
    cm  = confusion_matrix(y_test.argmax(axis=1), model.predict([X_test_features,X_test_masks]).argmax(axis=1))
    print(cm)
    report = classification_report(y_test.argmax(axis=1), model.predict([X_test_features,X_test_masks]).argmax(axis=1),zero_division=0 )
    print(report)
    



def random_noise_attack(features_mask, epsilon=0.2):
    noise = np.random.uniform(-epsilon, epsilon, features_mask.shape)
    adversarial_features_mask = features_mask + noise
    return np.clip(adversarial_features_mask, -2.0, 2.0)

def frequency_domain_attack(features_mask):
    # Convert feature masks to the frequency domain using Discrete Fourier Transform (DFT)
    features_mask_freq_domain = np.fft.fft2(features_mask)

    # Modify frequency components to introduce adversarial perturbations
    # Example: Amplify high-frequency components
    amplified_freq_domain = features_mask_freq_domain * 10

    # Convert back to the spatial domain using Inverse Discrete Fourier Transform (IDFT)
    adversarial_features_mask = np.fft.ifft2(amplified_freq_domain).real
    return adversarial_features_mask

def attack_train_test():
    model = tf.keras.models.load_model('../models/GRU-RNN')



    X_train_features = np.load('../data/numpyarrs/X_train_features.npy')
    X_train_masks = np.load('../data/numpyarrs/X_train_masks.npy')
    y_train = np.load('../data/numpyarrs/y_train.npy')


    X_adv_masks = random_noise_attack(X_train_masks)
    print(X_adv_masks.shape,X_train_masks.shape)

    _, accuracy = model.evaluate([X_train_features,X_adv_masks], y_train)
    print(f"""
           Test_loss : {round(_*100,2)}%
           Test accuracy : {round(accuracy * 100, 2)}%            
""")

    X_adv_masks = frequency_domain_attack(X_train_masks)
    _, accuracy = model.evaluate([X_train_features,X_adv_masks], y_train)
    print(f"""
           Test_loss : {round(_*100,2)}%
           Test accuracy : {round(accuracy * 100, 2)}%            
""")
    
    print("Confusion martix  adverisal training ")
    cm  = confusion_matrix(y_train.argmax(axis=1), model.predict([X_train_features,X_adv_masks]).argmax(axis=1))

    print(cm)

    report = classification_report(y_train.argmax(axis=1), model.predict([X_train_features,X_adv_masks]).argmax(axis=1),zero_division=0 )

    print(report)

    model.fit([X_train_features,X_adv_masks],y_train,validation_split=0.3,
        epochs=EPOCHS,batch_size=BATCH_SIZE,
        callbacks=[checkpoint])

    X_test_features = np.load('../data/numpyarrs/X_test_features.npy')
    X_test_masks = np.load('../data/numpyarrs/X_test_masks.npy')
    y_test = np.load('../data/numpyarrs/y_test.npy')

    X_adv_masks = random_noise_attack(X_test_masks)
    # print(X_adv_masks.shape,X_train_masks.shape)

    _, accuracy = model.evaluate([X_test_features,X_adv_masks], y_test)
    print(f"""
           Test_loss : {round(_*100,2)}%
           Test accuracy : {round(accuracy * 100, 2)}%            
""")

    X_adv_masks = frequency_domain_attack(X_test_masks)
    _, accuracy = model.evaluate([X_test_features,X_adv_masks], y_test)
    print(f"""
           Test_loss : {round(_*100,2)}%
           Test accuracy : {round(accuracy * 100, 2)}%            
""")
    print("Confusion martix after adverisal training ")
    print(X_test_features.shape,X_adv_masks.shape,y_test.shape)
    cm  = confusion_matrix(y_test.argmax(axis=1), model.predict([X_test_features,X_adv_masks]).argmax(axis=1))

    print(cm)

    report = classification_report(y_test.argmax(axis=1), model.predict([X_test_features,X_adv_masks]).argmax(axis=1) )

    print(report)


    X_adv_masks = random_noise_attack(X_train_masks)
    model.fit([X_train_features,X_adv_masks],y_train,validation_split=0.3,
        epochs=EPOCHS,batch_size=BATCH_SIZE,
        callbacks=[checkpoint])
    X_adv_masks = random_noise_attack(X_test_masks)
    # print(X_adv_masks.shape,X_train_masks.shape)

    _, accuracy = model.evaluate([X_test_features,X_adv_masks], y_test)
    print(f"""
           Test_loss : {round(_*100,2)}%
           Test accuracy : {round(accuracy * 100, 2)}%            
""")

    X_adv_masks = frequency_domain_attack(X_test_masks)
    _, accuracy = model.evaluate([X_test_features,X_adv_masks], y_test)
    print(f"""
           Test_loss : {round(_*100,2)}%
           Test accuracy : {round(accuracy * 100, 2)}%            
""")
    
    model.save("../models/CNN_RNN_Adv")






os.system('date')
## feature_extration
# feature_extractor = build_cnn_feature_extractor()
# feature_extractor.summary()

# X_train,y_train = prepare_all_videos(df, "../data/videos")
# X_test,y_test = prepare_all_videos(df2,'../data/videos')
# print(f"Frame features in train set: {X_train[0].shape}")
# print(f"Frame masks in train set: {X_train[1].shape}")
# print("Saving the data")

# #Saving data
# np.save('../data/numpyarrs/X_train_features.npy', X_train[0])
# np.save('../data/numpyarrs/X_train_masks.npy', X_train[1])
# np.save('../data/numpyarrs/y_train.npy', y_train)
# np.save('../data/numpyarrs/X_test_features.npy', X_test[0])
# np.save('../data/numpyarrs/X_test_masks.npy', X_test[1])
# np.save('../data/numpyarrs/y_test.npy', y_test)

## traning and testing 
# trainandtest(my_gru())

## Adverisal training and testing 
# attack_train_test()

## llms




os.system('date')
