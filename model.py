import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt

# 1. Veri Yükleme ve Hazırlık
data_dir = "datathon-ai-qualification-round/train"  # Tüm görüntülerin bulunduğu ana klasör
train_csv = "datathon-ai-qualification-round/train_data.csv"  # CSV dosyasının yolu

# CSV'yi yükle
df = pd.read_csv(train_csv)
print(df.head())

# Görüntülerin yollarını oluştur
df['file_path'] = df['filename'].apply(lambda x: os.path.join(data_dir, x))

# Sınıf etiketlerini sayısallaştır
labels = df['city'].unique()
label_to_index = {label: idx for idx, label in enumerate(labels)}
df['class_index'] = df['city'].map(label_to_index)
# class_index sütununu string formatına çevir
df['class_index'] = df['class_index'].astype(str)
# Veriyi train ve validation setlerine ayır
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class_index'], random_state=42)

# 2. Data Augmentation (Veri Çoğaltma)
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

# Generatorları oluştur
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='file_path',
    y_col='class_index',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    val_df,
    x_col='file_path',
    y_col='class_index',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# 3. Modelin Hazırlanması
# ResNet50 önceden eğitilmiş modelini yükle
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Yeni katmanlar ekleyin
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(labels), activation='softmax')(x)

# Modeli oluştur
model = Model(inputs=base_model.input, outputs=predictions)

# İlk katmanları dondur
for layer in base_model.layers:
    layer.trainable = False

# Modeli derle
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 4. Model Eğitimi
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    validation_steps=val_generator.n // val_generator.batch_size
)

# 5. Performans Görselleştirme
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show()

# 6. Modeli Kaydetme
model.save('resnet50_city_classifier.h5')