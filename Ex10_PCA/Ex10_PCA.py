import os
import numpy as np
import pandas as pd
import random
import shutil
import matplotlib.pyplot as plt
from cv2 import imread, imshow
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

dirs_df = pd.DataFrame(columns=['paths'])
files_df = pd.DataFrame(columns=['paths', 'root', 'label'])
dataset_path = r".\faces94"
classes = os.listdir(dataset_path)
img = []


for root, dirs, files in os.walk(dataset_path, topdown=True):
    for dir_name in dirs:
        dirs_df = dirs_df.append({'paths': os.path.join(root, dir_name)},
                                 ignore_index=True)

        # Choose a random file in each directory
        random_file = random.choice(os.listdir(str(dirs_df.paths.iloc[-1])))
        random_file_path = os.path.join(root, dir_name, random_file)

        if os.path.isfile(random_file_path):

            # Add file path to files df:
            files_df = files_df.append({'paths': random_file_path, 'root': root}, ignore_index=True)
            if classes[0] in random_file_path:
                files_df['label'].iloc[-1] = classes[0]
            elif classes[1] in random_file_path:
                files_df['label'].iloc[-1] = classes[1]

            # Copy files to new directory:
            shutil.copy(random_file_path, r'./random_files')

            # Read the file, and add it to the dataset
            img.append(imread(random_file_path, 0).reshape(-1, 1))  # 0 - Read as Grayscale


labels = files_df['label'].to_numpy()
dataset = np.column_stack(img)

# Normalized the data, and fit a pca
standard_normalize = StandardScaler()
standard_normalize.fit(dataset)
dataset_normalized = standard_normalize.transform(dataset)

pca = PCA(n_components=0.9, svd_solver='full')
pca.fit(dataset_normalized)
dataset_pca = pca.transform(dataset_normalized)

# Plot picture after PCA
for i in range(36):
    plt.subplot(6, 6, i+1)
    eigen_picture = dataset_pca[:, i].reshape((200, 180))
    plt.axis('off')
    plt.imshow(eigen_picture)

plt.figure(2)
plt.bar(range(0, len(pca.explaind_variance_ratio_)), height=pca.explained_variance_ratio_*100)
plt.xlabel('Principal Component')
plt.ylabel('% of Explained Variance')
plt.title('Scree Plot')
plt.show()

# My PCA:
a = (dataset_normalized.T@dataset_normalized)/len(dataset_normalized)
a_u, a_s, a_vh = np.linalg.svd(a)
