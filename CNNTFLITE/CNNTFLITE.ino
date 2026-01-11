#include <Arduino.h>
#include "int8.h"
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/tflite_bridge/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <Servo.h>

#define NUM_CLASSES 10
#define TENSOR_ARENA_SIZE 10 * 1024

// TensorFlow Lite globals
namespace {
  tflite::ErrorReporter* error_reporter = nullptr;
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
  TfLiteTensor* output = nullptr;
  
  // Tensor arena for model
  constexpr int kTensorArenaSize = TENSOR_ARENA_SIZE;
  uint8_t tensor_arena[kTensorArenaSize];
}

Servo servo1, servo2, servo3, servo4, servo5;
int servoPins[] = {2, 3, 4, 5, 6};

int pos = 1;
  float min_wl = 5104.5; 
  float max_wl = 7670.625; 
  float min_rms = 187.16344987334554; 
  float max_rms = 223.02737041085896; 
  float min_mav = 172.39791666666667; 
  float max_mav = 215.16354166666667; 
  float min_amp = 134.75; 
  float max_amp = 245.0;
  
// Fungsi normalisasi ke skala 0-255
float normalizeTo255(float value, float min, float max) { 
  return ((value - min) / (max - min)) * 255.0;
}

// Buffer untuk menyimpan data sementara 
const int numChannels = 8;
const int maxSets = 30;
int channelData[numChannels][maxSets]; 
int dataCount[numChannels] = {0};

// Variabel fitur untuk per channel 
float mavValues[numChannels];
float rmsValues[numChannels]; 
float wlValues[numChannels];
float amplitudeFirstBurst[numChannels];

// Fungsi untuk menggerakkan servo
void moveServosToPosition(int pos1, int pos2, int pos3, int pos4, int pos5) {
  servo1.write(pos1);
  servo2.write(pos2);
  servo3.write(pos3);
  servo4.write(pos4);
  servo5.write(pos5);
  delay(1000);
}

// Fungsi untuk menghitung Mean Absolute Value (MAV) 
float calculateMAV(int data[], int length) {
  float sumOfAbsoluteValues = 0.0; 
  for (int i = 0; i < length; i++) {
    sumOfAbsoluteValues += abs(data[i]);
  }
  return sumOfAbsoluteValues / length;
}

// Fungsi untuk menghitung Root Mean Square (RMS) 
float calculateRMS(int data[], int length) {
  float sumOfSquares = 0.0;
  for (int i = 0; i < length; i++) { 
    sumOfSquares += data[i] * data[i];
  }
  return sqrt(sumOfSquares / length);
}

// Fungsi untuk menghitung Waveform Length (WL)
float calculateWaveformLength(int data[], int length) { 
  float waveformLength = 0.0;
  for (int i = 1; i < length; i++) {
    waveformLength += abs(data[i] - data[i - 1]);
  }
  return waveformLength;
}

// Fungsi untuk menghitung Amplitudo Burst Pertama
float calculateAmplitudeFirstBurst(int data[], int length, float threshold) {
  for (int i = 0; i < length; i++) { 
    if (abs(data[i]) > threshold) {
      return abs(data[i]);
    }
  }
  return 0.0;
}

// Fungsi untuk mendapatkan pointer ke servo berdasarkan indeks
Servo* getServo(int index) { 
  switch (index) {
    case 0: return &servo1; 
    case 1: return &servo2; 
    case 2: return &servo3; 
    case 3: return &servo4; 
    case 4: return &servo5;
    default: return &servo1;
  }
}

void displayServoPositions() {
  Serial.print("Servo 1 Posisi: ");
  Serial.println(servo1.read());
  Serial.print("Servo 2 Posisi: "); 
  Serial.println(servo2.read());
  Serial.print("Servo 3 Posisi: "); 
  Serial.println(servo3.read());
  Serial.print("Servo 4 Posisi: "); 
  Serial.println(servo4.read());
  Serial.print("Servo 5 Posisi: ");
  Serial.println(servo5.read());
}

void performAction(int actionIndex) { 
  switch (actionIndex) {
    case 0: 
      Serial.println("Open");
      moveServosToPosition(0, 0, 0, 0, 0); 
      break;
    case 1:
      Serial.println("Close"); 
      moveServosToPosition(180, 180, 180, 180, 180); 
      break;
    case 2: 
      Serial.println("Finegrip");
      moveServosToPosition(180, 145, 0, 0, 0); 
      break;
    case 3: 
      Serial.println("Pointer");
      moveServosToPosition(180, 0, 180, 180, 160); 
      break;
    case 4: 
      Serial.println("Agree");
      moveServosToPosition(0, 160, 180, 180, 160); 
      break;
    case 5: 
      Serial.println("Two");
      moveServosToPosition(180, 0, 0, 180, 160); 
      break;
    case 6: 
      Serial.println("Three");
      moveServosToPosition(180, 0, 0, 0, 160); 
      break;
    case 7: 
      Serial.println("Four");
      moveServosToPosition(180, 0, 0, 0, 0); 
      break;
    case 8: 
      Serial.println("Pinch");
      moveServosToPosition(160, 140, 100, 180, 160); 
      break;
    case 9: 
      Serial.println("Halfclose");
      moveServosToPosition(160, 140, 110, 180, 120); 
      break;
    default: 
      Serial.println("Open");
      moveServosToPosition(0, 0, 0, 0, 0);
      break;
  }
}

void setup() {
  Serial.begin(115200);
  Serial1.begin(9600);
  Serial.println("Memulai pengumpulan data.. (Arduino Nano 33 BLE Sense)");

  // Inisialisasi servo
  servo1.attach(servoPins[0]);
  servo2.attach(servoPins[1]);
  servo3.attach(servoPins[2]);
  servo4.attach(servoPins[3]);
  servo5.attach(servoPins[4]);
  moveServosToPosition(0, 0, 0, 0, 0);

  // Setup TensorFlow Lite
  static tflite::MicroErrorReporter micro_error_reporter;
  error_reporter = &micro_error_reporter;

  // Load model
  model = tflite::GetModel(irisModel);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("ERROR: Model schema version mismatch!");
    Serial.print("Model version: ");
    Serial.println(model->version());
    Serial.print("Expected version: ");
    Serial.println(TFLITE_SCHEMA_VERSION);
    while (true);
  }

  Serial.println("Model loaded successfully!");

  // Create ops resolver
  static tflite::AllOpsResolver resolver;

  // Build interpreter
  static tflite::MicroInterpreter static_interpreter(
    model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  // Allocate memory for tensors
  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    Serial.println("ERROR: AllocateTensors() failed!");
    while (true);
  }

  // Get input and output tensors
  input = interpreter->input(0);
  output = interpreter->output(0);

  // Print model info
  Serial.print("Number of inputs: ");
  Serial.println(interpreter->inputs_size());
  Serial.print("Input shape: ");
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();
  
  Serial.print("Number of outputs: ");
  Serial.println(interpreter->outputs_size());
  Serial.print("Output shape: ");
  for (int i = 0; i < output->dims->size; i++) {
    Serial.print(output->dims->data[i]);
    Serial.print(" ");
  }
  Serial.println();

  Serial.print("Tensor Arena Size: ");
  Serial.println(kTensorArenaSize);
  Serial.println("Model berhasil dimuat ke Arduino Nano 33 BLE Sense!");
}

void loop() {
  if (Serial1.available() >= numChannels) {
    unsigned long startTime = millis();
    for (int i = 0; i < numChannels; i++) {
      if (dataCount[i] < maxSets) {
        int data = Serial1.read(); 
        channelData[i][dataCount[i]] = data;
        dataCount[i]++;
      }
    }

    // Cek apakah semua channel sudah penuh
    bool allChannelsFull = true;
    for (int i = 0; i < numChannels; i++) {
      if (dataCount[i] < maxSets) {
        allChannelsFull = false;
        break;
      }
    }

    if (allChannelsFull) {
      float avgMAV = 0, avgRMS = 0, avgWL = 0, avgAmpFirstBurst = 0;

      for (int i = 0; i < numChannels; i++) {
        mavValues[i] = calculateMAV(channelData[i], maxSets);
        rmsValues[i] = calculateRMS(channelData[i], maxSets);
        wlValues[i] = calculateWaveformLength(channelData[i], maxSets);
        amplitudeFirstBurst[i] = calculateAmplitudeFirstBurst(channelData[i], maxSets, 0.1);
        avgMAV += mavValues[i];
        avgRMS += rmsValues[i];
        avgWL += wlValues[i];
        avgAmpFirstBurst += amplitudeFirstBurst[i];
        dataCount[i] = 0;
      }

      avgMAV /= numChannels;
      avgRMS /= numChannels;
      avgWL /= numChannels;
      avgAmpFirstBurst /= numChannels;

      // Buat input untuk CNN
      float x1[12];
      x1[0] = 68.811792;
      for (int i = 1; i < 8; i++) {
        x1[i] = channelData[i][maxSets - 1];
      }
      x1[8] = avgWL;
      x1[9] = avgRMS;
      x1[10] = avgMAV;
      x1[11] = avgAmpFirstBurst;

      // Normalisasi ke 0-255
      float normalizedInput[12];
      for (int i = 0; i < 8; i++) {
        normalizedInput[i] = (x1[i] / 255.0) * 255.0;
      }
      normalizedInput[8]  = normalizeTo255(x1[8],  min_wl, max_wl);
      normalizedInput[9]  = normalizeTo255(x1[9],  min_rms, max_rms);
      normalizedInput[10] = normalizeTo255(x1[10], min_mav, max_mav);
      normalizedInput[11] = normalizeTo255(x1[11], min_amp, max_amp);

      // Debugging input
      Serial.println("=== Data Input ===");
      for (int i = 0; i < 12; i++) {
        Serial.print("x1["); Serial.print(i); Serial.print("] = ");
        Serial.println(x1[i]);
      }

      Serial.println("=== Data Setelah Normalisasi ===");
      for (int i = 0; i < 12; i++) {
        Serial.print("normalizedInput["); Serial.print(i); Serial.print("] = ");
        Serial.println(normalizedInput[i]);
      }

      // Copy normalized input to TFLite input tensor
      for (int i = 0; i < 12; i++) {
        input->data.f[i] = normalizedInput[i];
      }

      // Run inference
      TfLiteStatus invoke_status = interpreter->Invoke();
      if (invoke_status != kTfLiteOk) {
        Serial.println("ERROR: Invoke failed!");
        return;
      }

      // Get prediction results
      Serial.println("=== Skor Kelas ===");
      int predictedClass = 0;
      float maxScore = output->data.f[0];
      
      for (int i = 0; i < NUM_CLASSES; i++) {
        float score = output->data.f[i];
        Serial.print("Class "); Serial.print(i); Serial.print(": ");
        Serial.println(score, 4);
        
        if (score > maxScore) {
          maxScore = score;
          predictedClass = i;
        }
      }

      Serial.print("Predicted class: ");
      Serial.println(predictedClass);

      unsigned long endTime = millis();
      unsigned long duration = endTime - startTime;
      Serial.print("Durasi waktu loop: "); 
      Serial.print(duration);
      Serial.println(" ms");

      // Jalankan aksi servo
      performAction(predictedClass);
      displayServoPositions();

      delay(500);
    }
  }

  delay(100);
}