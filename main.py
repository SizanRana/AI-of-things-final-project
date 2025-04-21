import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd

# STEP 1: Path to data folder with TouchPose .pkl files
data_dir = './touch_pose'
file_names = [f'P{str(i).zfill(2)}.pkl' for i in range(1, 11)]

# STEP 2: Loading each .pkl file which contains a DataFrame
all_cap = []
all_depth = []
all_labels = []

for file in file_names:
    file_path = os.path.join(data_dir, file)
    print(f'Loading {file_path}...')
    with open(file_path, 'rb') as f:
        df = pickle.load(f, encoding='latin1')

        cap_imgs = np.stack(df['cap_img'].values)
        depth_imgs = np.stack(df['depth_img'].values)
        labels = df['gesture_id'].values

        all_cap.append(cap_imgs)
        all_depth.append(depth_imgs)
        all_labels.append(labels)

# STEP 3: Combining everything
X_cap = np.concatenate(all_cap)       
X_depth = np.concatenate(all_depth)   
y = np.concatenate(all_labels)

print("Data shapes:")
print("Capacitance:", X_cap.shape)
print("Depth:", X_depth.shape)
print("Labels:", y.shape)

# STEP 4: Sensor Fusion (flatten and combine cap + depth)
X_cap_flat = X_cap.reshape(X_cap.shape[0], -1)
X_depth_flat = X_depth.reshape(X_depth.shape[0], -1)
X_combined = np.concatenate([X_cap_flat, X_depth_flat], axis=1)

# STEP 5: Dimensionality Reduction
print("Applying PCA to reduce feature size...")
pca = PCA(n_components=100)
X_reduced = pca.fit_transform(X_combined)

# STEP 6: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# STEP 7: Train a classifier
print("Training Random Forest Classifier...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# STEP 8: Predict and Evaluate
y_pred = clf.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# STEP 9: Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()
