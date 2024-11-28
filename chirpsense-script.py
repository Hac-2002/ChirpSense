import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, 
    Concatenate, Layer, Reshape
)
from tensorflow.keras.utils import load_img, img_to_array, Sequence
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Enable memory growth for GPU
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# Configuration
class Config:
    def __init__(self):
        # Base path - change this to your data directory
        self.base_path = "./data"
        
        # Feature folders
        self.mel_visual_folder = os.path.join(self.base_path, "Mel_Spectogram/visualizations")
        self.zcr_visual_folder = os.path.join(self.base_path, "Zero_Crossing_Rate/visualizations")
        self.centroid_visual_folder = os.path.join(self.base_path, "Spectral_Centroid/visualizations")
        self.mfcc_visual_folder = os.path.join(self.base_path, "MFCC/visualizations")
        
        self.mel_numpy_folder = os.path.join(self.base_path, "Mel_Spectogram/numpy_features")
        self.zcr_numpy_folder = os.path.join(self.base_path, "Zero_Crossing_Rate/numpy_features")
        self.centroid_numpy_folder = os.path.join(self.base_path, "Spectral_Centroid/numpy_features")
        self.mfcc_numpy_folder = os.path.join(self.base_path, "MFCC/numpy_features")
        
        # Input shapes for images
        self.input_shape_mel = (234, 572, 3)
        self.input_shape_zcr = (235, 616, 3)
        self.input_shape_centroid = (234, 572, 3)
        self.input_shape_mfcc = (234, 572, 3)
        
        # Input shapes for numpy features
        self.input_shape_mel_np = (128, 1292)
        self.input_shape_zcr_np = (1, 1292)
        self.input_shape_centroid_np = (1, 1292)
        self.input_shape_mfcc_np = (20, 1292)
        
        self.num_classes = 264

class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, mel_paths, zcr_paths, centroid_paths, mfcc_paths,
                 mel_np_paths, zcr_np_paths, centroid_np_paths, mfcc_np_paths,
                 labels, config, batch_size=4, shuffle=True):
        self.config = config
        self.mel_paths = mel_paths
        self.zcr_paths = zcr_paths
        self.centroid_paths = centroid_paths
        self.mfcc_paths = mfcc_paths
        
        self.mel_np_paths = mel_np_paths
        self.zcr_np_paths = zcr_np_paths
        self.centroid_np_paths = centroid_np_paths
        self.mfcc_np_paths = mfcc_np_paths
        
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.mel_paths))
        
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.mel_paths) / self.batch_size))

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = min((idx + 1) * self.batch_size, len(self.mel_paths))
        batch_indexes = self.indexes[start_idx:end_idx]
        batch_size = len(batch_indexes)

        # Initialize arrays for images
        mel_imgs = np.zeros((batch_size,) + self.config.input_shape_mel, dtype=np.float32)
        zcr_imgs = np.zeros((batch_size,) + self.config.input_shape_zcr, dtype=np.float32)
        centroid_imgs = np.zeros((batch_size,) + self.config.input_shape_centroid, dtype=np.float32)
        mfcc_imgs = np.zeros((batch_size,) + self.config.input_shape_mfcc, dtype=np.float32)

        # Initialize arrays for numpy features
        mel_features = np.zeros((batch_size,) + self.config.input_shape_mel_np, dtype=np.float32)
        zcr_features = np.zeros((batch_size,) + self.config.input_shape_zcr_np, dtype=np.float32)
        centroid_features = np.zeros((batch_size,) + self.config.input_shape_centroid_np, dtype=np.float32)
        mfcc_features = np.zeros((batch_size,) + self.config.input_shape_mfcc_np, dtype=np.float32)

        batch_labels = np.zeros(batch_size, dtype=np.int32)

        for i, idx in enumerate(batch_indexes):
            try:
                # Load and preprocess images
                mel_img = load_img(self.mel_paths[idx], target_size=self.config.input_shape_mel[:2])
                zcr_img = load_img(self.zcr_paths[idx], target_size=self.config.input_shape_zcr[:2])
                centroid_img = load_img(self.centroid_paths[idx], target_size=self.config.input_shape_centroid[:2])
                mfcc_img = load_img(self.mfcc_paths[idx], target_size=self.config.input_shape_mfcc[:2])

                mel_imgs[i] = img_to_array(mel_img) / 255.0
                zcr_imgs[i] = img_to_array(zcr_img) / 255.0
                centroid_imgs[i] = img_to_array(centroid_img) / 255.0
                mfcc_imgs[i] = img_to_array(mfcc_img) / 255.0

                # Load numpy features
                mel_feat = np.load(self.mel_np_paths[idx])
                zcr_feat = np.load(self.zcr_np_paths[idx])
                centroid_feat = np.load(self.centroid_np_paths[idx])
                mfcc_feat = np.load(self.mfcc_np_paths[idx])

                # Ensure correct shapes
                if mel_feat.shape != self.config.input_shape_mel_np:
                    mel_feat = np.resize(mel_feat, self.config.input_shape_mel_np)
                if zcr_feat.shape != self.config.input_shape_zcr_np:
                    zcr_feat = np.resize(zcr_feat, self.config.input_shape_zcr_np)
                if centroid_feat.shape != self.config.input_shape_centroid_np:
                    centroid_feat = np.resize(centroid_feat, self.config.input_shape_centroid_np)
                if mfcc_feat.shape != self.config.input_shape_mfcc_np:
                    mfcc_feat = np.resize(mfcc_feat, self.config.input_shape_mfcc_np)

                mel_features[i] = mel_feat
                zcr_features[i] = zcr_feat
                centroid_features[i] = centroid_feat
                mfcc_features[i] = mfcc_feat

                batch_labels[i] = self.labels[idx]

            except Exception as e:
                print(f"Error loading batch item {i}, index {idx}: {str(e)}")
                continue

        return {
            "Mel_Image_Input": mel_imgs,
            "ZCR_Image_Input": zcr_imgs,
            "Centroid_Image_Input": centroid_imgs,
            "MFCC_Image_Input": mfcc_imgs,
            "Mel_Numpy_Input": mel_features,
            "ZCR_Numpy_Input": zcr_features,
            "Centroid_Numpy_Input": centroid_features,
            "MFCC_Numpy_Input": mfcc_features
        }, batch_labels

def build_efficient_model(config):
    """Build the multi-input neural network model"""
    
    # Input 1: Mel Spectrogram Images
    mel_input = Input(shape=config.input_shape_mel, name="Mel_Image_Input")
    x1 = Conv2D(32, (3, 3), activation='relu', padding='same')(mel_input)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = MaxPooling2D((2, 2))(x1)
    x1 = BatchNormalization()(x1)
    x1 = Flatten()(x1)
    x1 = Dense(256, activation='relu')(x1)
    x1 = Dropout(0.5)(x1)

    # Input 2: ZCR Images
    zcr_img_input = Input(shape=config.input_shape_zcr, name="ZCR_Image_Input")
    x2 = Conv2D(32, (3, 3), activation='relu', padding='same')(zcr_img_input)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = BatchNormalization()(x2)
    x2 = Flatten()(x2)
    x2 = Dense(256, activation='relu')(x2)
    x2 = Dropout(0.5)(x2)

    # Input 3: Spectral Centroid Images
    centroid_img_input = Input(shape=config.input_shape_centroid, name="Centroid_Image_Input")
    x3 = Conv2D(32, (3, 3), activation='relu', padding='same')(centroid_img_input)
    x3 = MaxPooling2D((2, 2))(x3)
    x3 = Conv2D(64, (3, 3), activation='relu', padding='same')(x3)
    x3 = MaxPooling2D((2, 2))(x3)
    x3 = BatchNormalization()(x3)
    x3 = Flatten()(x3)
    x3 = Dense(256, activation='relu')(x3)
    x3 = Dropout(0.5)(x3)

    # Input 4: MFCC Images
    mfcc_img_input = Input(shape=config.input_shape_mfcc, name="MFCC_Image_Input")
    x4 = Conv2D(32, (3, 3), activation='relu', padding='same')(mfcc_img_input)
    x4 = MaxPooling2D((2, 2))(x4)
    x4 = Conv2D(64, (3, 3), activation='relu', padding='same')(x4)
    x4 = MaxPooling2D((2, 2))(x4)
    x4 = BatchNormalization()(x4)
    x4 = Flatten()(x4)
    x4 = Dense(256, activation='relu')(x4)
    x4 = Dropout(0.5)(x4)

    # Input 5: Mel Spectrogram Numpy Features
    mel_np_input = Input(shape=config.input_shape_mel_np, name="Mel_Numpy_Input")
    x5 = Dense(512, activation='relu')(Flatten()(mel_np_input))
    x5 = BatchNormalization()(x5)
    x5 = Dropout(0.3)(x5)
    x5 = Dense(256, activation='relu')(x5)
    x5 = Dropout(0.3)(x5)

    # Input 6: ZCR Numpy Features
    zcr_np_input = Input(shape=config.input_shape_zcr_np, name="ZCR_Numpy_Input")
    x6 = Dense(256, activation='relu')(Flatten()(zcr_np_input))
    x6 = BatchNormalization()(x6)
    x6 = Dropout(0.3)(x6)

    # Input 7: Spectral Centroid Numpy Features
    centroid_np_input = Input(shape=config.input_shape_centroid_np, name="Centroid_Numpy_Input")
    x7 = Dense(256, activation='relu')(Flatten()(centroid_np_input))
    x7 = BatchNormalization()(x7)
    x7 = Dropout(0.3)(x7)

    # Input 8: MFCC Numpy Features
    mfcc_np_input = Input(shape=config.input_shape_mfcc_np, name="MFCC_Numpy_Input")
    x8 = Dense(512, activation='relu')(Flatten()(mfcc_np_input))
    x8 = BatchNormalization()(x8)
    x8 = Dropout(0.3)(x8)
    x8 = Dense(256, activation='relu')(x8)
    x8 = Dropout(0.3)(x8)

    # Concatenate all branches
    combined = Concatenate()([x1, x2, x3, x4, x5, x6, x7, x8])
    
    # Final dense layers
    x = Dense(1024, activation='relu')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(config.num_classes, activation='softmax', name="output")(x)

    # Create model
    model = Model(
        inputs={
            "Mel_Image_Input": mel_input,
            "ZCR_Image_Input": zcr_img_input,
            "Centroid_Image_Input": centroid_img_input,
            "MFCC_Image_Input": mfcc_img_input,
            "Mel_Numpy_Input": mel_np_input,
            "ZCR_Numpy_Input": zcr_np_input,
            "Centroid_Numpy_Input": centroid_np_input,
            "MFCC_Numpy_Input": mfcc_np_input
        },
        outputs=output
    )
    
    return model

def get_file_paths