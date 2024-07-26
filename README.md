# Persistent-Homology-in-Imaging-and-Finance
## PERSISTENT HOMOLOGY OF IMAGE
`import matplotlib.pyplot as plt
from skimage import color
from skimage import io, color`

#### loaded in variables: image1, image2, image3
`image1 = io.imread('Brain1.jfif')
image2 = io.imread('Brain2.jpg')
image3 = io.imread('Brain3.jpg')`

#### Convert images to grayscale
`gray_image1 = color.rgb2gray(image1)
gray_image2 = color.rgb2gray(image2)
gray_image3 = color.rgb2gray(image3)`

#### Display the images
`fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(gray_image1, cmap='gray')
ax[0].set_title('Brain Image 1')
ax[1].imshow(gray_image2, cmap='gray')
ax[1].set_title('Brain Image 2')
ax[2].imshow(gray_image3, cmap='gray')
ax[2].set_title('Brain Image 3')
plt.show()`

![image](https://github.com/user-attachments/assets/852869d3-3216-4d2e-a801-06008c8733d5)

## EXTRACT POINT CLOUD
`import matplotlib.pyplot as plt
from skimage import io, color, filters
import numpy as np`

#### Convert images to grayscale
`gray_image1 = color.rgb2gray(image1)
gray_image2 = color.rgb2gray(image2)
gray_image3 = color.rgb2gray(image3)`

#### Normalize images
`norm_image1 = (gray_image1 - np.min(gray_image1)) / (np.max(gray_image1) - np.min(gray_image1))
norm_image2 = (gray_image2 - np.min(gray_image2)) / (np.max(gray_image2) - np.min(gray_image2))
norm_image3 = (gray_image3 - np.min(gray_image3)) / (np.max(gray_image3) - np.min(gray_image3))`

#### Threshold the images to create binary images
`threshold1 = filters.threshold_otsu(norm_image1)
binary_image1 = norm_image1 > threshold1`

`threshold2 = filters.threshold_otsu(norm_image2)
binary_image2 = norm_image2 > threshold2
`
`threshold3 = filters.threshold_otsu(norm_image3)
binary_image3 = norm_image3 > threshold3`
#### Extract point clouds from the binary images
`points1 = np.column_stack(np.nonzero(binary_image1))
points2 = np.column_stack(np.nonzero(binary_image2))
points3 = np.column_stack(np.nonzero(binary_image3))`

#### Display the binary images with points
`fig, ax = plt.subplots(1, 3, figsize=(15, 5))
ax[0].imshow(binary_image1, cmap='gray')
ax[0].scatter(points1[:, 1], points1[:, 0], s=0.1, c='red')
ax[0].set_title('Brain Image 1 (Binary with Points)')
ax[1].imshow(binary_image2, cmap='gray')
ax[1].scatter(points2[:, 1], points2[:, 0], s=0.1, c='red')
ax[1].set_title('Brain Image 2 (Binary with Points)')
ax[2].imshow(binary_image3, cmap='gray')
ax[2].scatter(points3[:, 1], points3[:, 0], s=0.1, c='red')
ax[2].set_title('Brain Image 3 (Binary with Points)')
plt.show()`

#### Print shapes of the point clouds
`print(points1.shape, points2.shape, points3.shape)`

![image](https://github.com/user-attachments/assets/450acd1f-5129-4a8b-8a12-cd6ec88e9e6d)

## TO VISUALIZE THE 3D POINT CLOUD, PERSISTENT DIAGRAM, BARCODE AND BETTI CURVE
`import matplotlib.pyplot as plt`
`from mpl_toolkits.mplot3d import Axes3D`
`from skimage import io, color, filters, transform`
`import gudhi as gd`
`import numpy as np`
`from gudhi.wasserstein import wasserstein_distance`

#### Load and resize the images
`image1 = io.imread('Brain1.jfif')
image1 = transform.resize(image1, (image1.shape[0] // 2, image1.shape[1] // 2))`

#### Convert images to grayscale
`gray_image1 = color.rgb2gray(image1)`

#### Normalize images
`norm_image1 = (gray_image1 - np.min(gray_image1)) / (np.max(gray_image1) - np.min(gray_image1))`

#### Threshold the images to create binary images
`threshold1 = filters.threshold_otsu(norm_image1)
binary_image1 = norm_image1 > threshold1`

#### Extract point clouds from the binary images
`points1 = np.column_stack(np.nonzero(binary_image1))`

#### Sample points to reduce complexity
`if points1.shape[0] > 1000:
    indices = np.random.choice(points1.shape[0], 2000, replace=False)
    points1 = points1[indices]`

#### Display the binary images with points
`fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(binary_image1, cmap='gray')
ax.scatter(points1[:, 1], points1[:, 0], s=0.1, c='red')
ax.set_title('Brain Image 1 (Binary with Points)')
plt.show()`

#### Display the 3D scatter plot
`fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points1[:, 1], points1[:, 0], np.zeros(points1.shape[0]), s=0.1, c='red')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Scatter Plot of Point Cloud')
plt.show()`

#### Define functions to compute VR complex and persistence
`def compute_vr_complex(point_cloud, max_edge_length):
    rips_complex = gd.RipsComplex(points=point_cloud, max_edge_length=max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=2)
    return simplex_tree`

`def compute_persistence(simplex_tree):
    simplex_tree.persistence()
    return simplex_tree`

`def plot_persistence_diagram(simplex_tree):
    gd.plot_persistence_diagram(simplex_tree.persistence_intervals_in_dimension(0))
    gd.plot_persistence_diagram(simplex_tree.persistence_intervals_in_dimension(1))
    plt.show()  `

`def plot_barcode(simplex_tree):
    gd.plot_persistence_barcode(simplex_tree.persistence_intervals_in_dimension(0))
    gd.plot_persistence_barcode(simplex_tree.persistence_intervals_in_dimension(1))
    plt.show()`

`def plot_betti_curve(simplex_tree, max_edge_length):
    barcodes = simplex_tree.persistence_intervals_in_dimension(0)
    betti_numbers = np.zeros(int(max_edge_length) + 1)
    for interval in barcodes:
        start = int(interval[0])
        end = int(interval[1]) if interval[1] != np.inf else len(betti_numbers)
        betti_numbers[start:end] += 1`
    
 `   plt.plot(range(len(betti_numbers)), betti_numbers, label="Betti Curve")
    plt.xlabel("Filtration Value")
    plt.ylabel("Betti Number")
    plt.title("Betti Curve")
    plt.legend()
    plt.show()`

#### Example usage
`max_edge_length = 10.0  # Reduced based on data complexity
simplex_tree1 = compute_vr_complex(points1, max_edge_length)
simplex_tree1 = compute_persistence(simplex_tree1)`

#### Plot persistence diagram, barcode, and Betti curve
`1plot_persistence_diagram(simplex_tree1)
plot_barcode(simplex_tree1)
plot_betti_curve(simplex_tree1, max_edge_length)1`

#### Calculate Wasserstein distance between persistence diagrams of different dimensions (0 and 1)
`persistence_0 = simplex_tree1.persistence_intervals_in_dimension(0)
persistence_1 = simplex_tree1.persistence_intervals_in_dimension(1)`


![image](https://github.com/user-attachments/assets/f12756a7-ff09-4e74-962f-2e69b763dc46), ![image](https://github.com/user-attachments/assets/3c95f90f-3ff1-415d-8207-f41b02fe3b38)
![image](https://github.com/user-attachments/assets/a6298991-3e60-4e0e-9d18-f4c593dc0ea6), ![image](https://github.com/user-attachments/assets/0f5530c4-bc48-4769-93e3-d08e10d25675)
![image](https://github.com/user-attachments/assets/25c776d1-d0ef-4983-a390-d89b2d3e4376), ![image](https://github.com/user-attachments/assets/eae78a98-21ae-4901-8126-fe9629c4229d)










