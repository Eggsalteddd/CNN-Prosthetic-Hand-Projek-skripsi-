import tensorflow as tf

# Ganti path ke model kamu
model_path = r"Skripsi/int8.tflite"

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# ambil detail model
details = interpreter.get_tensor_details()
ops = interpreter._get_ops_details()  # private method tapi bisa dipanggil

print(f"Jumlah operator (TF_NUM_OPS): {len(ops)}\n")
print("Daftar operator:")
for i, op in enumerate(ops):
    print(f"{i+1}. {op['op_name']}")