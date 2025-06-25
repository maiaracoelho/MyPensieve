import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("✅ GPU detectada:")
    for gpu in gpus:
        print(" ->", gpu)
else:
    print("❌ Nenhuma GPU detectada. Verifique as configurações do Docker e TensorFlow.")
