import pandas as pd
from pathlib import Path

from keras.applications import mobilenet_v2, vgg19, resnet50, vgg16
from keras.layers import Dense, GlobalAveragePooling2D, Dropout #,Flatten
from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator

TEMPLATES = {
    'mobilenetv2' : (mobilenet_v2.MobileNetV2, mobilenet_v2.preprocess_input),
    'vgg16' : (vgg16.VGG16, vgg16.preprocess_input),
    'vgg19' : (vgg19.VGG19, vgg19.preprocess_input),
    'resnet50' : (resnet50.ResNet50, resnet50.preprocess_input)
}

def make_model(input_shape = (224,224,3), template='mobilenetv2', weights='imagenet'):
    # Load template
    try:
        base_template, preprocess_input = TEMPLATES[template]
    except:
        print('Template to be chosen among: '+ ','.join(TEMPLATES.keys())) 
    # Create base model (already trained) from template
    base = base_template( input_shape=input_shape,
                                include_top=False,
                                weights=weights)
    base.trainable = False
    # Create head
    head = Sequential([
                        GlobalAveragePooling2D(), Dropout(0.3),
                        Dense(128,activation='relu'), Dropout(0.3),
                        Dense(128,activation='relu'), Dropout(0.3),
                        Dense(32,activation='relu'), Dropout(0.3),
                        Dense(32,activation='relu'), Dropout(0.3),
                        Dense(1,activation='sigmoid')
                    ])
    # Full model
    model = Sequential([base, head])
    # Return result
    return model, preprocess_input


def read_info(csv_path):
    # Load base csv
    path = Path(csv_path)
    df = pd.read_csv(path)
    # Minor formatting
    df.id = df.id.apply(lambda s: path.parent.joinpath(s+'.jpg').resolve().as_posix())
    df.label = df.label.apply(lambda x: 'Parasite' if x else 'No Parasite')
    return df


def load_labels(base_folder, csv_names='ground_truth.csv'):
    # Find all csv files
    base_path = Path(base_folder)
    all_files = list(base_path.rglob(csv_names))
    # Load them all into one dataframe
    df = pd.concat(map(read_info,all_files))
    return df


def make_train_test(df, preprocess_input, sample=1, train_ratio= 0.8, **kwargs):
    image_datagen = ImageDataGenerator(#zoom_range=0.1,
                                        #width_shift_range=0.1,
                                        #height_shift_range=0.1,
                                        #rotation_range=360,   # Can cause distortions
                                        fill_mode = 'constant',
                                        cval = 0,
                                        preprocessing_function=preprocess_input,
                                        )

    # Shuffle and take a sample
    data = df.sample(frac=sample).reset_index(drop=True)  
    
    # Train/test split - OK to do it by position because df is already shuffled
    idx = int(len(data)*train_ratio)
    train_data = image_datagen.flow_from_dataframe(data.iloc[:idx], **kwargs)
    test_data = image_datagen.flow_from_dataframe(data.iloc[idx:], **kwargs)
    
    return train_data, test_data

