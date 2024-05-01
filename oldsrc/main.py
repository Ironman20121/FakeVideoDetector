
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from art.attacks.evasion import DeepFool, CarliniL2Method ,BasicIterativeMethod
from art.estimators.classification import TensorFlowV2Classifier
from tensorflow.keras.losses import BinaryCrossentropy
import os



class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self,epoch,logs={}):
        if (logs.get('val_accuracy')<0.2 and logs.get('val_accuracy') > 0.95):
            print('Stoping training as of now ')
            self.model.stop_training = True



class CustomModelCheckpoint(tf.keras.callbacks.Callback):
    def __init__(self, filepath, monitor1='val_loss', monitor2='val_accuracy', mode1='min', mode2='max', save_best_only=True, save_weights_only=True):
        super(CustomModelCheckpoint, self).__init__()
        self.filepath = filepath
        self.monitor1 = monitor1
        self.monitor2 = monitor2
        self.mode1 = mode1
        self.mode2 = mode2
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.best_loss = float('inf') if mode1 == 'min' else -float('inf')
        self.best_accuracy = 0.0

    def on_epoch_end(self, epoch, logs=None):
        current_loss = logs.get(self.monitor1)
        current_accuracy = logs.get(self.monitor2)

        if current_loss is None:
            print("Warning: Can't save best model weights based on validation loss as no validation loss is available.")
        elif current_accuracy is None:
            print("Warning: Can't save best model weights based on validation accuracy as no validation accuracy is available.")
        else:
            if (self.mode1 == 'min' and current_loss < self.best_loss) or (self.mode1 == 'max' and current_loss > self.best_loss):
                self.best_loss = current_loss
                if self.save_best_only:
                    self.model.save_weights(self.filepath, overwrite=True)
            if (self.mode2 == 'min' and current_accuracy < self.best_accuracy) or (self.mode2 == 'max' and current_accuracy > self.best_accuracy):
                self.best_accuracy = current_accuracy
                if not self.save_best_only:
                    self.model.save_weights(self.filepath, overwrite=True)


def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))



# Define the CNN model
def build_cnn_model(input_shape):
    # act_f ='LeakyReLU'
    # act_f = mish
    act_f='relu'

    model = tf.keras.models.Sequential([
    
        #Conv2D
        layers.Conv2D(32,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.5),
        

        layers.Conv2D(64,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),
       
        layers.Conv2D(128,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),



        layers.Conv2D(256,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),


       
        layers.Conv2D(512,(5,5),activation =act_f,input_shape=input_shape, padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.25),

        #Flattern 
        layers.Flatten(),
        layers.Dense(512,activation='LeakyReLU'),
        layers.Dropout(0.5),
        layers.Dense(256,activation='LeakyReLU'),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()
    return model





def build_vit_model(image_size):
    patch_size = 2
    num_patches = (image_size // patch_size) ** 2
    projection_dim = 32
    num_heads = 5
    transformer_layers = 10
    num_classes = 2

    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Patch embedding
    patches = layers.Reshape((-1, patch_size * patch_size * 3))(inputs)
    encoded_patches = layers.Dense(projection_dim, activation="relu")(patches)

    # Positional encoding
    position_embedding = layers.Embedding(input_dim=num_patches, output_dim=projection_dim)(tf.range(num_patches))
    encoded_patches_with_positions = encoded_patches + position_embedding

    # Transformer encoder
    for _ in range(transformer_layers):
        # Multi-head self-attention
        x1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=projection_dim)(encoded_patches_with_positions, encoded_patches_with_positions)
        # Add & Norm
        x1 = layers.Add()([encoded_patches_with_positions, x1])
        x1 = layers.LayerNormalization()(x1)
        # Feed forward
        x2 = layers.Dense(units=projection_dim, activation="relu")(x1)
        x2 = layers.Dense(units=projection_dim)(x2)
        # Add & Norm
        encoded_patches_with_positions = layers.Add()([x1, x2])
        encoded_patches_with_positions = layers.LayerNormalization()(encoded_patches_with_positions)

    # Classification head
    cls_token = layers.Lambda(lambda x: tf.math.reduce_mean(x, axis=1))(encoded_patches_with_positions)
    outputs = layers.Dense(num_classes, activation="sigmoid")(cls_token)

    # Create the model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)





    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])
    model.summary()

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




def custom_binary_accuracy(y_true, y_pred):
    y_pred_binary = tf.argmax(y_pred, axis=1)  # Convert softmax probabilities to class predictions
    y_true_binary = tf.cast(y_true, tf.int64)  # Ensure labels are integers
    accuracy = tf.reduce_mean(tf.cast(tf.equal(y_true_binary, y_pred_binary), tf.float32))
    return accuracy



def main():
    print(os.system('date'))
    print(tf.__version__)

    # Create the CNN model
    input_shape = (112,112, 3)  # Input shape of preprocessed video data
    # cnn_model = build_cnn_model(input_shape)
    # vit_model ,checkpoint_callback = build_vit_model(input_shape[0])
    
    
    # already saved in numpy arrays data load 
    X_train = np.load('data/X_train.npy')
    y_train = np.load('data/y_train.npy')
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')
    X_val = np.load('data/X_val.npy')
    y_val = np.load('data/y_val.npy')
    
    train_generator = (X_train,y_train)
    val_gen= (X_val,y_val)
    test_gen = (X_test,y_test)
    # train_img_path = 'data/train_img'
    # test_img_path = 'data/test_img'
    # val_img_path = 'data/val_img'
    # datagen = ImageDataGenerator(
    #     rescale=1./255,
    #     rotation_range=20,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    #     shear_range=0.2,
    #     zoom_range=0.2,
    #     horizontal_flip=True,
    #     vertical_flip=True,
    #     fill_mode='nearest')
   
    # train_generator = datagen.flow_from_directory(
    #     train_img_path,
    #     target_size=(input_shape[0], input_shape[1]),
    #     batch_size=32,
    #     class_mode='categorical',
    #     shuffle=True)
    
    # test_gen = datagen.flow_from_directory(
    #     test_img_path,
    #     target_size=(input_shape[0], input_shape[1]),
    #     batch_size=32,
    #     class_mode='categorical',
    #     shuffle=False

    # )
    # val_gen = datagen.flow_from_directory(
    #     val_img_path,
    #     target_size=(input_shape[0], input_shape[1]),
    #     batch_size=32,
    #     class_mode='categorical',
    #     shuffle=False

    # )

    # Train the model
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_accuracy', factor=0.2, patience=5, min_lr=0.00001)
    # early_stop = tf.keras.callbacks.EarlyStopping(
    # monitor='val_loss',
    # patience=3,
    # verbose=1,
    # mode='min',
    # )
    checkpoint_filepath = 'models/best_weights.h5'
    checkpoint_callback = CustomModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor1='val_loss',  # Monitor validation loss
    monitor2='val_accuracy',  # Monitor validation accuracy
    mode1='min',  # Mode for validation loss
    mode2='max',  # Mode for validation accuracy
    save_best_only=True,  # Save only the best weights
    save_weights_only=True  # Save only the weights, not the entire model
)
    checkpoint_callback_2 = myCallback()

    # print(f"X_train: {X_train.shape},y_train :{y_train.shape}")
    # cnn_model.fit(X_train,y_train,epochs=100,batch_size=64,callbacks=[checkpoint_callback,reduce_lr,checkpoint_callback_2],validation_data=(X_val,y_val))
    # cnn_model.fit(train_generator, epochs=100, batch_size=32,steps_per_epoch=100,callbacks=[checkpoint_callback,reduce_lr,checkpoint_callback_2],validation_data=val_gen)
    # cnn_model = tf.keras.models.load_model('models/cnn')
    # saving model for future use 
    # cnn_model.save('models/cnn_check')
    
    
#     # Evaluate the model
#     print(f"X_train: {X_train.shape},y_train :{y_train.shape}")
#     cnn_model = tf.keras.models.load_model('models/cnn_check')
#     # cnn_model.load_weights('models/best_weights.h5')
#     # cnn_model.save('models/cnn_check')
    
    
#     train_loss, train_accuracy = cnn_model.evaluate(X_train,y_train)
#     print(f"""
#            Training_loss : {train_loss}
#            Test Accuracy : {train_accuracy}             
# """)
#     print(f"X_test: {X_test.shape},y_test :{y_test.shape}")
#     test_loss, test_accuracy = cnn_model.evaluate(X_test,y_test)

#     print("Test Loss:", test_loss)
#     print("Test Accuracy:", test_accuracy)

#     print(f"""
#             X_val:{X_val.shape}
#             y_val:{y_val.shape}
# """)
#     val_loss,val_acc = cnn_model.evaluate(X_val,y_val)
#     print("val Loss:", val_loss)
#     print("val Accuracy:", val_acc)
    
    cnn_model = tf.keras.models.load_model('models/cnn-deepfool')
    ##Adverisal attack 
    num_frames = X_test.shape[1]
    ##Convert your TensorFlow model to an ART classifier
    art_classifier = TensorFlowV2Classifier(model=cnn_model, clip_values=(0.0, 1.0), input_shape=(num_frames, 112, 112, 3), nb_classes=2)
    
    ##Define the loss function
    loss_object = BinaryCrossentropy(from_logits=False)

    ##Convert your TensorFlow model to an ART classifier
    art_classifier = TensorFlowV2Classifier(
    model=cnn_model,
    clip_values=(0.0, 1.0),
    input_shape=(num_frames, 112, 112, 3),
    nb_classes=2,
    loss_object=loss_object  # Provide the loss function here
    )

    ##Generate adversarial examples for test it on validation video data
    ##DeepFool_Attack
    deepfool_eg = adversarial_attack(DeepFool(art_classifier), X_val, y_val,cnn_model)
    # BasicIterativeMethod Attack
    BIM_eg = adversarial_attack(BasicIterativeMethod(art_classifier), X_val, y_val,cnn_model)
    # CarliniL2Method  
  
    carliniL2Method_eg = adversarial_attack(CarliniL2Method(art_classifier), X_val, y_val,cnn_model)
    
#     ##Deep fool training 
#     deepfool_adversarial_training = cnn_model.fit(deepfool_eg,y_val,epochs=100, batch_size=64, validation_data=(X_test,y_test),callbacks=[checkpoint_callback,checkpoint_callback_2,reduce_lr])
#     ##Check after deep fool
#     adversarial_attack(DeepFool(art_classifier), X_test, y_test,cnn_model)
#     adversarial_attack(BasicIterativeMethod(art_classifier), X_test, y_test,cnn_model)
#     adversarial_attack(CarliniL2Method(art_classifier), X_test, y_test,cnn_model)
#     cnn_model.load_weights('models/best_weights.h5')
#     cnn_model.save("models/cnn-deepfool")
    
    ##Carlini training
 
    carlini_adversarial_training = cnn_model.fit(carliniL2Method_eg,epochs=100, batch_size=64, validation_data=(X_test,y_test),callbacks=[checkpoint_callback,checkpoint_callback_2,reduce_lr])
    adversarial_attack(DeepFool(art_classifier), X_test, y_test,cnn_model)
    BIM_eg = adversarial_attack(BasicIterativeMethod(art_classifier), X_test, y_test,cnn_model)
    adversarial_attack(CarliniL2Method(art_classifier), X_test, y_test,cnn_model)
    cnn_model.load_weights('models/best_weights.h5')
    cnn_model.save("models/cnn-carlini")

    BIM_adversarial_training = cnn_model.fit(BIM_eg,y_val,epochs=100, batch_size=64, validation_data=(X_test,y_test),callbacks=[checkpoint_callback,checkpoint_callback_2,reduce_lr])
    adversarial_attack(DeepFool(art_classifier), X_test, y_test,cnn_model)
    adversarial_attack(BasicIterativeMethod(art_classifier), X_test, y_test,cnn_model)
    adversarial_attack(CarliniL2Method(art_classifier), X_test, y_test,cnn_model)
    cnn_model.load_weights('models/best_weights.h5')
    cnn_model.save("models/cnn-patch")
    print(os.system('date'))




main()




