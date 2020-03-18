from pathlib import Path
from train_utils import make_model, read_info, load_labels, make_train_test
from keras.callbacks import TensorBoard, CSVLogger,  EarlyStopping, ModelCheckpoint

######  Global variables
######  TEMPORARY - to be replaced by comman line arguments
DATA_DIR = Path('../../../../').joinpath('datasets','thick_smears_150_x1')
EPOCHS = 20
KWARGS = dict(  x_col='id',
                y_col='label',
                class_mode='binary',
                target_size=(224,224),
                batch_size=128,   #careful not to be too big for memory
                seed=2020)

######  Train script

# Load model and labels
model, preprocess_input = make_model()
labels = load_labels(DATA_DIR)
# Create dataset
train_data, test_data = make_train_test(labels, preprocess_input,
                                        sample=0.01, train_ratio= 0.9, **KWARGS)
# Set-up callbacks
callbacks = [
            CSVLogger('training_log.csv', append=True),
            EarlyStopping(monitor='val_acc', patience=2, restore_best_weights=True),
            ModelCheckpoint(filepath='temp.h5', monitor='val_acc')
            ]
# Training loop
model.compile(optimizer='adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
model.fit_generator(generator=train_data,
                    validation_data=test_data,
                    epochs=EPOCHS,
                    callbacks=callbacks,
                    )
model.save('trained_model.h5')
