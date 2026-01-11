  #include "int8.h" // Ganti dengan model Anda
  #include <tflm_esp32.h>
  #include <eloquent_tinyml.h>
  #include <ESP32Servo.h>


  #define ARENA_SIZE 15000 // memory arena size for the model
  #define RXD2 18
  #define TXD2 17

  // Definisi jumlah kelas untuk CNN 
  #define NUM_CLASSES 10

  Eloquent::TF::Sequential<TF_NUM_OPS, ARENA_SIZE> tf; 
  Servo servo1, servo2, servo3, servo4, servo5;
  int servoPins[] = {9, 10, 11, 12, 13};	// Pin servo 

  int pos = 1;	// Posisi servo

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

  // Fungsi untuk menggerakkan servo ke posisi tertentu
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
    return 0.0; // Jika tidak ada burst yang melebihi threshold
  }

  // Fungsi untuk mendapatkan pointer ke servo berdasarkan indeks
  Servo* getServo(int index) { 
    switch (index) {
      case 0: return &servo1; 
      case 1: return &servo2; 
      case 2: return &servo3; 
      case 3: return &servo4; 
      case 4: return &servo5;
      default: return &servo1; // Default ke servo1 jika terjadi kesalahan
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
   case 1:
      Serial.println("Close");
      moveServosToPosition(150, 150, 150, 150, 150);
      break;
    case 2:
      Serial.println("Finegrip");
      moveServosToPosition(150, 120, 0, 0, 0);
      break;
    case 3:
      Serial.println("Pointer");
      moveServosToPosition(150, 0, 150, 150, 150);
      break;
    case 4: 
      Serial.println("Agree");
      moveServosToPosition(0, 130, 150, 150, 150); 
      break;
    case 5: 
      Serial.println("Two");
      moveServosToPosition(150, 0, 0, 150, 100); 
      break;
    case 6: 
      Serial.println("Three");
      moveServosToPosition(150, 0, 0, 0, 130); 
      break;
    case 7: 
      Serial.println("Four");
      moveServosToPosition(150, 0, 0, 0, 0); 
      break;
    case 8: 
      Serial.println("Pinch");
      moveServosToPosition(150, 150, 150, 40, 40); //tengah 110
      break;
    case 9: 
      Serial.println("Halfclose");
      moveServosToPosition(90, 80, 130, 130, 90); //tengah 100
      default: Serial.println("Open");
        moveServosToPosition(0, 0, 0, 0, 0);	// Posisi default
        break;
      }
    }

  void setup() {
    Serial.begin(115200);
    Serial2.begin(9600, SERIAL_8N1, RXD2, TXD2);
    Serial.println("Memulai pengumpulan data.. (ESP32-S3)");


    // Inisialisasi servo
    for (int i = 0; i < 5; i++) {
      Servo* currentServo = getServo(i);
      currentServo->attach(servoPins[i]);
      currentServo->write(pos);
    }

    // Inisialisasi model
    tf.setNumInputs(1 * 12 * 1);
    tf.setNumOutputs(NUM_CLASSES);

    tf.resolver.AddQuantize();      // OP 1
    tf.resolver.AddExpandDims();    // OP. 2,5,8,11,14
    tf.resolver.AddConv2D();        // OP 3,6,12
    tf.resolver.AddReshape();       // OP 4,7,10,13,16,20
    tf.resolver.AddMaxPool2D();     // OP 9,15
    tf.resolver.AddShape();         // OP 17
    tf.resolver.AddStridedSlice();  // OP 18
    tf.resolver.AddPack();          // OP 19
    tf.resolver.AddFullyConnected(); // OP 21,22
    tf.resolver.AddSoftmax();       // OP 23
    tf.resolver.AddDequantize();    // OP 24

  while (!tf.begin(irisModel).isOk()) {
    Serial.print("Error load model: ");
    Serial.println(tf.exception.toString());
    delay(1000);
  }

  Serial.println("Model berhasil dimuat ke ESP32-S3!");
}
  void loop() {
    if (Serial2.available() >= numChannels) {
    unsigned long startTime = millis();
      for (int i = 0; i < numChannels; i++) { 
        if (dataCount[i] < maxSets) {
          int data = Serial2.read(); 
          channelData[i][dataCount[i]] = data; 
          dataCount[i]++;
        }
    }
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
        avgAmpFirstBurst +=amplitudeFirstBurst[i];
        dataCount[i] = 0;
    }

    avgMAV /= numChannels; 
    avgRMS /= numChannels;
    avgWL /= numChannels; 
    avgAmpFirstBurst /= numChannels;
    
    float x1[12]; 
    x1[0] = 68.811792; 
    for (int i = 1; i < 8; i++) { 
      x1[i] = channelData[i][maxSets - 1]; 
    } 
    x1[8] = avgWL; 
    x1[9] = avgRMS; 
    x1[10] = avgMAV; 
    x1[11] = avgAmpFirstBurst;

    // Normalisasi data input 
    float normalizedInput[12]; 
    for (int i = 0; i < 8; i++) {
      normalizedInput[i] = (x1[i] / 255.0) * 255;
    }
    normalizedInput[8] = normalizeTo255(x1[8], min_wl, max_wl);
    normalizedInput[9] = normalizeTo255(x1[9], min_rms, max_rms);
    normalizedInput[10] = normalizeTo255(x1[10], min_mav, max_mav);
    normalizedInput[11] = normalizeTo255(x1[11], min_amp, max_amp);

    // Cetak data input dan data setelah dinormalisasi
    Serial.println("Data Input:"); 
    for (int i = 0; i < 12; i++) {
      Serial.print("x1["); 
      Serial.print(i); 
      Serial.print("]: "); 
      Serial.println(x1[i]);
    }

    Serial.println("Data Setelah Normalisasi:"); 
    for (int i = 0; i < 12; i++) {
      Serial.print("normalizedInput["); 
      Serial.print(i);
      Serial.print("]: "); 
      Serial.println(normalizedInput[i]);
    }
    // Prediksi dengan model CNN 
    if(!tf.predict((float*)normalizedInput).isOk())  {
      Serial.println(tf.exception.toString()); 
      return;
    }

    // Dapatkan prediksi kelas dan skor
    int predictedClass = tf.classification; 
    float scores[NUM_CLASSES];

    // Cetak skor untuk setiap kelas
    Serial.println("Skor Prediksi untuk SetiapKelas:");
    for (int i = 0; i < NUM_CLASSES; i++) { 
      Serial.print("Class "); 
      Serial.print(i);
      Serial.print(": "); 
      Serial.println(tf.output(i), 4);
    }
    
    // Cetak kelas yang diprediksi 
    Serial.print("Predicted class: "); 
    Serial.println(predictedClass);
    
    unsigned long endTime = millis();
    unsigned long duration = endTime - startTime;
    // Durasi dalam milidetik
    // Tampilkan durasi loop di Serial Monitor 
    Serial.print("Durasi waktu loop: "); Serial.print(duration);
    Serial.println(" ms");
    
    performAction(predictedClass);	// Gerakkan servo berdasarkan hasil CNN
    
    displayServoPositions();
    
    delay(50);
    }
  }  
    delay(10);
  }