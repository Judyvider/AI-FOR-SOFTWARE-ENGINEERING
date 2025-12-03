import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os

def run_edge_ai_prototype():
    print("--- Task 1: Edge AI Prototype (Recyclable Item Classification) ---")
    
    # 1. Dataset Preparation
    # We use Fashion MNIST as a proxy for recyclable items (0=T-shirt, 1=Trouser, etc.)
    # In a real scenario, you would replace this with a dataset of 'Plastic', 'Glass', 'Paper'.
    print("\n[Step 1] Loading and Preprocessing Dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.fashion_mnist.load_data()
    
    # Normalize pixel values to 0-1 range
    train_images = train_images.astype('float32') / 255.0
    test_images = test_images.astype('float32') / 255.0
    
    # Reshape for CNN input (28, 28, 1)
    train_images = train_images.reshape((60000, 28, 28, 1))
    test_images = test_images.reshape((10000, 28, 28, 1))
    
    print(f"Training Samples: {len(train_images)}")
    print(f"Test Samples: {len(test_images)}")

    # 2. Define a Lightweight Model (MobileNet-style simplified architecture)
    # Edge models need fewer parameters.
    print("\n[Step 2] Building Lightweight CNN Model...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10) # 10 Output classes
    ])
    
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])

    # 3. Train Model
    # Short training for demonstration (1 epoch)
    print("\n[Step 3] Training Model...")
    model.fit(train_images, train_labels, epochs=1, validation_split=0.1)

    # 4. Convert to TensorFlow Lite (The "Edge" Step)
    print("\n[Step 4] Converting to TensorFlow Lite with Quantization...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Enable default optimizations (Quantization) to reduce size and latency
    # This converts weights from float32 to int8/float16 where possible
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    tflite_model = converter.convert()

    # Save the .tflite file
    tflite_filename = 'recycling_model_quantized.tflite'
    with open(tflite_filename, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = os.path.getsize(tflite_filename) / 1024
    print(f"Success! Model saved to disk.")
    print(f"Final Edge Model Size: {size_kb:.2f} KB (Optimized for IoT Devices)")

    # 5. Simulate Edge Inference
    # This part mimics what runs on the Raspberry Pi
    print("\n[Step 5] Simulating On-Device Inference...")
    
    # Load the interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Run inference on a single test image
    input_shape = input_details[0]['shape']
    # Use the first image from test set
    input_data = np.expand_dims(test_images[0], axis=0) 
    
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    prediction = np.argmax(output_data)
    
    print(f"Test Image True Label: {test_labels[0]}")
    print(f"Edge Model Prediction: {prediction}")
    print("Inference complete.")

if __name__ == "__main__":
    run_edge_ai_prototype()