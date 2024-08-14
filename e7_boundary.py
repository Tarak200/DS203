import cv2
import os
from google.colab.patches import cv2_imshow
import numpy as np

import zipfile

def unzip_and_save(zip_file_path, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract files from the zip archive
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(output_dir)

    print("Files extracted successfully.")

# Example usage
zip_file_path = "/content/E7-images.zip"
output_dir = "/content/extracted_files"

unzip_and_save(zip_file_path, output_dir)

def find_minimum_area_rectangle(image_path,length,breadth,filenames):
  # Read the image in grayscale
  img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

  # Apply thresholding to isolate the white region (adjust threshold as needed)
  thresh = 127  # Experiment with this value
  ret, thresh_img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY_INV)

  # Find contours
  contours, _ = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  #print(len(contours))
  if contours:
    # Assuming there's only one white region
    cnt = contours[0]

    # Approximate the contour with a polygon
    approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

    # Find the convex hull of the polygon (tightest fitting shape)
    hull = cv2.convexHull(approx)

    # Find the minimum area rectangle for the convex hull
    rect = cv2.minAreaRect(hull)
    length.append(max(rect[1]))
    breadth.append(min(rect[1]))
    print(image_path[-17:-13])
    filenames.append(image_path[-17:-13])
    # Get the corners of the rectangle
    box = cv2.boxPoints(rect)

    # Convert the points to integer coordinates (using recommended approach)
    box = box.astype(np.intp)

    # Draw the rectangle on the image
    img = cv2.drawContours(img, [box], 0, (0, 255,0), 2)  # Green rectangle

    # Show the image
    cv2_imshow(img)
    result_path = image_path[:-13] + "_tightbox.jpg"
    cv2.imwrite(result_path, img)
  else:
    print("No contours found")

def edge_only_img(image_path,length,breadth,filenames):
  # Read the image
  img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

  # Apply edge detection
  edges = cv2.Canny(img, 100, 200)

  # Invert the edges
  edges = cv2.bitwise_not(edges)

  # Threshold the inverted edges to get boundary line
  ret, thresh = cv2.threshold(edges, 127, 255, cv2.THRESH_BINARY)

  # Make everything black in the original image
  result = np.zeros_like(img)

  # Assign white color (255) to the boundary line in the original image
  result[thresh == 255] = 255

  # Display the result
  cv2_imshow(result)
  result_path = image_path[:-4] + "_boundary.jpg"
  cv2.imwrite(result_path, result)

  return find_minimum_area_rectangle(result_path,length,breadth,filenames)

# Path to the directory containing files
directory_path = "/content/extracted_files"
length =[]
breadth = []
filenames = []
# List all files in the directory
files = os.listdir(directory_path)

count = 0
for file in files:
  file_path = os.path.join(directory_path,file)
  print(file_path)
  edge_only_img(file_path,length,breadth,filenames)
  count = count+1
  # if(count==2):
  #   break

from PIL import Image
import os

# Function to count non-white pixels in an image
def count_non_white_pixels(image_path):
    print(image_path[-8:-4])
    image = Image.open(image_path)
    width, height = image.size
    non_white_count = 0

    for y in range(height):
        for x in range(width):
            pixel = image.getpixel((x, y))
            if pixel != (255, 255, 255):  # Check if pixel is not white
                non_white_count += 1
    total_area[image_path[-8:-4]] = non_white_count
    return non_white_count

# Directory containing images
images_directory = "/content/extracted_files"

# Initialize total area counter
total_area = {}

# Iterate over each image in the directory
count = 0
for filename in os.listdir(images_directory):
  if 'boundary' not in filename:
    if 'tbox' not in filename:
        if filename.endswith(".jpg"):
          image_path = os.path.join(images_directory, filename)
          non_white_pixels = count_non_white_pixels(image_path)
          count = count+1
          # if(count==5):
      #   break

print("Total area of non-white pixels:", total_area)

area_rect = []
for i in range(len(length)):
  area_rec = length[i]*breadth[i]
  area_rect.append(int(area_rec))
print(area_rect)

print(range(len(length)))
print(range(len(breadth)))
print(breadth)
print(length)
print(filenames)
print(total_area)

area_in_white = []
area_figure = []
aspectratio = []
bbratio = []
for i in range(len(length)):
  aspectratio.append(length[i]/breadth[i])
  area_figure.append(total_area[filenames[i]])
  area1 = total_area[filenames[i]]
  area2 = area_rect[i]
  bbratio.append( area_figure[i]/area_rect[i])
  area_in_white.append(int(area2 - area1))
print(area_in_white)

features = []
for i in range(len(length)):
  features.append([length[i],breadth[i],area_figure[i], area_rect[i],aspectratio[i],bbratio[i]])

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

no_clusters = 3
kmeans = KMeans(n_clusters=no_clusters)

# Fit the KMeans model to the scaled features
kmeans.fit(features_scaled)

# Get the cluster labels assigned to each data point
cluster_labels = kmeans.labels_

for i in range(len(length)):
  print(filenames[i], ' : ',cluster_labels[i])
# knn_classifier = KNeighborsClassifier(n_neighbors=5)
# knn_classifier.fit(features_scaled, labels)

threecluster = {1:[],0:[],2:[]}
for i in range(len(length)):
  threecluster[int(cluster_labels[i])].append(filenames[i])

print('Cluster 0 :',len(threecluster[0]))
print('Cluster 1 :',len(threecluster[1]))
print('Cluster 2 :',len(threecluster[2]))



from sklearn.metrics import silhouette_score

# Assuming you have a features array
# features = ...

# Initialize an empty list to store silhouette scores for each feature
silhouette_scores = []
features = np.array(features)
# Iterate over each feature
for i in range(features.shape[1]):
    # Create a new features array with only one feature
    single_feature = features[:, i].reshape(-1, 1)

    # Initialize KMeans with the number of clusters
    kmeans = KMeans(n_clusters=3)

    # Fit KMeans to the single feature
    kmeans.fit(single_feature)

    # Calculate silhouette score for the single feature
    silhouette = silhouette_score(single_feature, kmeans.labels_)

    # Append silhouette score to the list
    silhouette_scores.append(silhouette)
featu =['length','breadth', 'Area of shape','Area of tight-fit box','Aspect ratio','Bounding box ratio']
# Print silhouette scores for each feature
for i, score in enumerate(silhouette_scores):
    print(f"Feature- {featu[i]}: Silhouette Score = {score}")

from sklearn.metrics import silhouette_samples
from sklearn.preprocessing import MinMaxScaler

# Assuming you have a features array and already performed clustering
# features = ...
# cluster_labels = ...

# Convert features to a numpy array
features = np.array(features)

# Initialize KMeans with the number of clusters
kmeans = KMeans(n_clusters=3)

# Fit KMeans to the features
kmeans.fit(features)

# Calculate silhouette scores for each data point
silhouette_values = silhouette_samples(features, kmeans.labels_)

# Normalize silhouette scores to range from 0 to 1
scaler = MinMaxScaler()
silhouette_normalized = scaler.fit_transform(silhouette_values.reshape(-1, 1)).flatten()

# Print normalized silhouette scores for each data point Cluster = {cluster_labels[i]},
for i, score in enumerate(silhouette_normalized):
    print(f"Data Point {filenames[i]}: Silhouette Score = {score:.2f}, Complexity Score = {(1-score):.2f}")

import matplotlib.pyplot as plt

# Assuming you have performed clustering and have cluster labels
# cluster_labels = ...

# Plot clusters in 2D (assuming you have two significant features)
plt.figure(figsize=(8, 6))
plt.scatter(features[:, 0], features[:, 1], c=cluster_labels, cmap='viridis', s=50, alpha=0.5)
plt.title('KMeans Clustering')
plt.xlabel('Length')
plt.ylabel('Breadth')
plt.colorbar(label='Cluster')
plt.show()

import seaborn as sns
import pandas as pd
# Assuming you have performed clustering and have cluster labels
# features, cluster_labels = ...

# Assuming you have already performed KMeans and have the cluster centers
# kmeans = ...
# cluster_centers = kmeans.cluster_centers_

# Plot clusters in 2D (using the first two features)
sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=cluster_labels, palette='viridis')
plt.title('Scatter Plot of Clusters')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()
# Plot cluster centers in 2D (using the first two features)
sns.scatterplot(x=features[:, 0], y=features[:, 1], hue=cluster_labels, palette='viridis')
cluster_centers = kmeans.cluster_centers_
sns.scatterplot(x=cluster_centers[:, 0], y=cluster_centers[:, 1], color='red', marker='X', s=100, label='Cluster Centers')
plt.title('Cluster Centers')
plt.xlabel('Length')
plt.ylabel('Breadth')
plt.legend()
plt.show()
df_features = pd.DataFrame(features)

# Flatten the cluster_labels array
cluster_labels_flat = cluster_labels.flatten()
# Check the shapes of the arrays
print("Features shape:", features.shape)
print("Cluster labels shape:", cluster_labels_flat.shape)
# Pairwise feature scatter plots
# sns.pairplot(df_features, diag_kind='kde', hue=cluster_labels_flat, palette='viridis')
plt.title('Pairwise Feature Scatter Plots')
plt.show()
from sklearn.metrics import silhouette_samples

# Calculate silhouette scores for each data point
silhouette_values = silhouette_samples(features, cluster_labels)

# Plot silhouette scores sorted by cluster labels
sns.boxplot(x=cluster_labels, y=silhouette_values, palette='viridis')
plt.title('Silhouette Plot')
plt.xlabel('Cluster')
plt.ylabel('Silhouette Score')
plt.show()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming you have performed clustering and have cluster labels
# features, cluster_labels = ...

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
pca_result = pca.fit_transform(features)

# Plot PCA results
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=cluster_labels, cmap='viridis')
plt.title('PCA Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.colorbar(label='Cluster')
plt.show()

from sklearn.manifold import TSNE

# Assuming you have performed clustering and have cluster labels
# features, cluster_labels = ...

# Apply t-SNE for dimensionality reduction
tsne = TSNE(n_components=2, random_state=42)
tsne_result = tsne.fit_transform(features)

# Plot t-SNE results
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=cluster_labels, cmap='viridis')
plt.title('t-SNE Visualization')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Cluster')
plt.show()

import scipy.cluster.hierarchy as shc

# Assuming you have performed hierarchical clustering and have cluster labels
# features, cluster_labels = ...

# Calculate linkage matrix
linkage_matrix = shc.linkage(features, method='ward')

# Plot dendrogram
plt.figure(figsize=(10, 6))
dendrogram = shc.dendrogram(linkage_matrix)
plt.title('Dendrogram')
plt.xlabel('Samples')
plt.ylabel('Distance')
plt.show()

