import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.optimizers import SGD
import tensorflow as tf
import os

# =====================================
# Load Dataset
# =====================================
df = pd.read_excel(r"D:\Code\Skripsi\datasetbaru.xlsx")

# =====================================
# Feature Extraction Functions
# =====================================
def calculate_wl(values):
    """Waveform Length (WL)"""
    return np.sum(np.abs(np.diff(values)))

def calculate_rms(values):
    """Root Mean Square (RMS)"""
    return np.sqrt(np.mean(np.square(values)))

def calculate_mav(values):
    """Mean Absolute Value (MAV)"""
    return np.mean(np.abs(values))

def calculate_amplitude_first_burst(values, threshold=0.1):
    """Amplitude of First Burst"""
    burst_indices = np.where(np.abs(values) > threshold)[0]
    if len(burst_indices) > 0:
        return np.max(np.abs(values[burst_indices[0]:burst_indices[0] + 1]))
    return 0  # Jika tidak ada burst yang terdeteksi, kembalikan 0

# =====================================
# Feature Extraction per Class & Label
# =====================================
data = df.copy()

for class_label in df['Class'].unique():
    for label in df['Label'].unique():
        filtered_data = df[(df['Class'] == class_label) & (df['Label'] == label)]
        if filtered_data.empty:
            continue

        # Ambil data dari 8 channel
        channel_data = filtered_data[[
            'Channel1', 'Channel2', 'Channel3', 'Channel4',
            'Channel5', 'Channel6', 'Channel7', 'Channel8'
        ]]

        # Hitung fitur per channel
        wl_values_per_channel = channel_data.apply(calculate_wl, axis=0)
        rms_values_per_channel = channel_data.apply(calculate_rms, axis=0)
        mav_values_per_channel = channel_data.apply(calculate_mav, axis=0)
        amplitude_first_burst_per_channel = channel_data.apply(calculate_amplitude_first_burst, axis=0)

        # Hitung rata-rata fitur
        wl_average = wl_values_per_channel.mean()
        rms_average = rms_values_per_channel.mean()
        mav_average = mav_values_per_channel.mean()
        amplitude_first_burst_average = amplitude_first_burst_per_channel.mean()

        # Tambahkan hasil fitur ke DataFrame
        df.loc[filtered_data.index, 'WL'] = wl_average
        df.loc[filtered_data.index, 'RMS'] = rms_average
        df.loc[filtered_data.index, 'MAV'] = mav_average
        df.loc[filtered_data.index, 'Amplitude_First_Burst'] = amplitude_first_burst_average

# =====================================
# Persiapan Dataset untuk CNN
# =====================================
print(df.head(100))

# Pisahkan fitur dan label
features = df.drop(columns=["Label", "Class"])
Class = df["Class"]

print(Class.unique())

# Standarisasi (Z-score)
scaler = StandardScaler()
X = scaler.fit_transform(features)

# One-hot encoding
y = to_categorical(Class)

# Normalisasi ke 0–255
X = (X - X.min()) / (X.max() - X.min()) * 255

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ubah bentuk untuk CNN (samples, timesteps, channels)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

# =====================================
# Build CNN Model
# =====================================
model = Sequential()

# 1st Conv Layer
model.add(Conv1D(16, 3, padding='same', input_shape=(12, 1)))  # 12 fitur input
model.add(BatchNormalization())
model.add(Activation('relu'))

# 2nd Conv Layer
model.add(Conv1D(64, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

# 3rd Conv Layer
model.add(Conv1D(32, 3, padding='same'))
model.add(BatchNormalization())
model.add(Activation('relu'))
model.add(MaxPooling1D(pool_size=2))

# Fully Connected Layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))  # 10 kelas output

# Compile Model
opt = SGD(learning_rate=0.1, momentum=0.9)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

# Train Model
history = model.fit(X_train, y_train, epochs=100, batch_size=128, validation_split=0.2)

# Summary
model.summary()

# =====================================
# Evaluation
# =====================================
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {accuracy * 100:.2f}%')

# --- Training Data ---
train_predictions = model.predict(X_train)
y_pred_train_classes = np.argmax(train_predictions, axis=1)
y_true_train_classes = np.argmax(y_train, axis=1)

conf_matrix_train = confusion_matrix(y_true_train_classes, y_pred_train_classes)
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Training Data')
plt.show()

print(classification_report(y_true_train_classes, y_pred_train_classes))
print("Training Accuracy:", accuracy_score(y_true_train_classes, y_pred_train_classes))

# --- Test Data ---
test_predictions = model.predict(X_test)
y_pred_test_classes = np.argmax(test_predictions, axis=1)
y_true_test_classes = np.argmax(y_test, axis=1)

conf_matrix_test = confusion_matrix(y_true_test_classes, y_pred_test_classes)
sns.heatmap(conf_matrix_test, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - Test Data')
plt.show()

print(classification_report(y_true_test_classes, y_pred_test_classes))
print("Test Accuracy:", accuracy_score(y_true_test_classes, y_pred_test_classes))

# =====================================
# Save Keras Model
# =====================================
model_save_path = r"D:\Code\Skripsi\emg_cnn_model.h5"
model.save(model_save_path)
print(f"Model saved to {model_save_path}")

# =====================================
# Plot Training Loss dan Accuracy
# =====================================
plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid()

plt.show()


# =====================================
# Convert Keras Model to TensorFlow Lite with INT8 Quantization
# =====================================

# Load model
loaded_model = tf.keras.models.load_model(model_save_path)

# Representative dataset generator untuk kalibrasi INT8
def representative_dataset():
    """
    Generator yang memberikan sampel data untuk kalibrasi quantisasi INT8
    """
    num_calibration_steps = 100  # Jumlah sampel untuk kalibrasi
    for i in range(min(num_calibration_steps, len(X_train))):
        yield [X_train[i:i+1].astype(np.float32)]

# Konversi ke TFLite dengan INT8 Quantization
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

# Set optimasi untuk INT8 quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Set representative dataset untuk kalibrasi
converter.representative_dataset = representative_dataset

# Untuk full integer quantization (input dan output juga INT8)
# Uncomment baris berikut jika ingin input/output juga INT8:
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model_int8 = converter.convert()

# Simpan model TFLite INT8
tflite_save_path_int8 = r"D:\Code\Skripsi\int8.tflite"
""
with open(tflite_save_path_int8, "wb") as f:
    f.write(tflite_model_int8)

print(f"\nINT8 Quantized TFLite model saved to {tflite_save_path_int8}")

# Cek ukuran file
original_size = os.path.getsize(model_save_path) / 1024  # KB
quantized_size = os.path.getsize(tflite_save_path_int8) / 1024  # KB
print(f"\nOriginal Keras model size: {original_size:.2f} KB")
print(f"INT8 Quantized TFLite model size: {quantized_size:.2f} KB")
print(f"Size reduction: {((original_size - quantized_size) / original_size * 100):.2f}%")

# =====================================
# Verifikasi Akurasi Model INT8
# =====================================
print("\n" + "="*50)
print("Verifying INT8 Quantized Model Accuracy")
print("="*50)

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=tflite_save_path_int8)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print(f"Input shape: {input_details[0]['shape']}")
print(f"Input type: {input_details[0]['dtype']}")
print(f"Output shape: {output_details[0]['shape']}")
print(f"Output type: {output_details[0]['dtype']}")

# Test pada semua data test
correct = 0
y_pred_int8 = []
y_true_int8 = []

input_scale, input_zero_point = input_details[0]['quantization']

print("Input scale:", input_scale)
print("Input zero point:", input_zero_point)

for i in range(len(X_test)):
    # Ambil data float (0–255)
    input_data = X_test[i:i+1]

    # Quantize ke INT8 (WAJIB)
    input_data_int8 = (input_data / input_scale + input_zero_point).astype(np.int8)

    interpreter.set_tensor(
        input_details[0]['index'],
        input_data_int8
    )
    # Run inference
    interpreter.invoke()
    
    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output_data)
    true_class = np.argmax(y_test[i])
    
    y_pred_int8.append(predicted_class)
    y_true_int8.append(true_class)
    
    if predicted_class == true_class:
        correct += 1

accuracy_int8 = correct / len(X_test) * 100
print(f"\nINT8 Quantized Model Test Accuracy: {accuracy_int8:.2f}%")
print(f"Original Model Test Accuracy: {accuracy * 100:.2f}%")
print(f"Accuracy difference: {abs(accuracy * 100 - accuracy_int8):.2f}%")

# Confusion Matrix untuk INT8 Model
conf_matrix_int8 = confusion_matrix(y_true_int8, y_pred_int8)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_int8, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix - INT8 Quantized Model (Test Data)')
plt.show()

print("\nClassification Report - INT8 Quantized Model:")
print(classification_report(y_true_int8, y_pred_int8))