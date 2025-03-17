import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import tensorflow as tf


csv_file = "asl_keypoints.csv" 
df = pd.read_csv(csv_file)


df.dropna(inplace=True)


X = df.iloc[:, 1:-1].values  # Exclude 'frame' column and label
y = df.iloc[:, -1].values    # Gesture label column


scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

y_onehot = tf.keras.utils.to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)

np.savez("asl_preprocessed_data.npz", X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test)

print("✅ Data preprocessing complete! Saved as 'asl_preprocessed_data.npz'")
print(f"Training samples: {X_train.shape[0]}, Testing samples: {X_test.shape[0]}")
