import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import mlflow
import mlflow.tensorflow

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models

# ================= CONFIG =================
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 10

TRAIN_DIR = "chest_xray/train"
VAL_DIR = "chest_xray/val"
TEST_DIR = "chest_xray/test"

MODEL_PATH = "models/final_model.keras"
BEST_FILE = "best_accuracy.txt"

# ============== HELPER FUNCTIONS =================
def get_best_accuracy():
    if os.path.exists(BEST_FILE):
        with open(BEST_FILE, "r") as f:
            return float(f.read())
    return 0.0

def save_best_accuracy(acc):
    with open(BEST_FILE, "w") as f:
        f.write(str(acc))

# ============== MLFLOW =================
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("Pneumonia Detection CNN")

# ============== DATA =================
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=15,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical"
)

test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

NUM_CLASSES = len(train_gen.class_indices)
print("Classes detected:", train_gen.class_indices)

# ============== MODEL =================
base_model = EfficientNetB0(
    weights="imagenet",
    include_top=False,
    input_shape=(IMG_SIZE, IMG_SIZE, 3)
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.4),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True
    ),
    tf.keras.callbacks.ModelCheckpoint(
        "models/best_model.keras",
        save_best_only=True
    )
]

class_weight = {
    0: 1.0,
    1: 0.8,
    2: 1.0
}

# ============== TRAINING =================
with mlflow.start_run(run_name="efficientnet_finetune"):

    # Log params
    mlflow.log_param("img_size", IMG_SIZE)
    mlflow.log_param("batch_size", BATCH_SIZE)
    mlflow.log_param("epochs", EPOCHS)

    # Initial training
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks,
        class_weight=class_weight
    )

    print("Starting Fine-Tuning...")

    # Load best weights
    model.load_weights("models/best_model.keras")

    # Unfreeze top layers
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # Fine-tuning
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        class_weight=class_weight
    )

    # ============== EVALUATION =================
    loss, acc = model.evaluate(test_gen)
    print(f"Test Accuracy: {acc*100:.2f}%")

    # Log metric
    mlflow.log_metric("test_accuracy", acc)

    # ============== COMPARISON =================
    best_acc = get_best_accuracy()

    print(f"Best Previous Accuracy: {best_acc}")
    print(f"Current Accuracy: {acc}")

    # ============== CONDITIONAL SAVE =================
    if acc > best_acc:
        print("✅ New model is better. Saving...")

        model.save(MODEL_PATH)
        save_best_accuracy(acc)

        mlflow.tensorflow.log_model(model, "pneumonia_model")

        # CI/CD flag
        with open("model_improved.txt", "w") as f:
            f.write("YES")

    else:
        print("❌ Model not better. Skipping save.")

        with open("model_improved.txt", "w") as f:
            f.write("NO")