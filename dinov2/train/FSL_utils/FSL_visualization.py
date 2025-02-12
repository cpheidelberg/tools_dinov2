
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import torch
import seaborn as sns
from sklearn.metrics import confusion_matrix

def cohen_kappa_with_plot(predicted, true, path = None):
    """
    Calculate Cohen's Kappa for two arrays and plot the confusion matrix.
    
    Parameters:
    predicted (array-like): Array of predicted labels
    true (array-like): Array of true labels
    
    Returns:
    float: Cohen's Kappa score
    """
    # Ensure inputs are numpy arrays
    predicted = np.array(predicted)
    true = np.array(true)
    
    # Compute the confusion matrix
    cm = confusion_matrix(true, predicted, normalize=True)

  #  print(cm)
    # Number of observations
    n = np.sum(cm)
    
    # Observed agreement
    P_o = np.trace(cm) / n
    
    # Expected agreement
    sum_rows = np.sum(cm, axis=1)
    sum_cols = np.sum(cm, axis=0)
    P_e = np.sum(sum_rows * sum_cols) / (n * n)
    
    # Cohen's Kappa
    kappa = (P_o - P_e) / (1 - P_e)
   # print(kappa)
    # Plotting the confusion matrix
  #  plt.figure(figsize=(8, 6))
  #  sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, )
  #  plt.xlabel('Predicted Labels')
  #  plt.ylabel('True Labels')
  #  plt.title('Confusion Matrix')


    if path:
        plt.savefig(path, dpi = 200)

 #   plt.show()
    
    return kappa

def normalize_confusion_matrix(conf_matrix, axis=1):
    """
    Normalizes a confusion matrix along the specified axis.

    Parameters:
    conf_matrix (np.ndarray): The confusion matrix to normalize.
    axis (int): The axis to normalize along. 1 for row-wise (default), 0 for column-wise.

    Returns:
    np.ndarray: The normalized confusion matrix.
    """
    if axis == 1:
        # Normalize each row (true class counts)
        row_sums = conf_matrix.sum(axis=1, keepdims=True)
        normalized_matrix = np.round(conf_matrix / row_sums,2)
    elif axis == 0:
        # Normalize each column (predicted class counts)
        col_sums = conf_matrix.sum(axis=0, keepdims=True)
        normalized_matrix = conf_matrix / col_sums
    else:
        raise ValueError("Axis must be 0 (column-wise) or 1 (row-wise).")

    # Replace any NaNs resulting from division by zero with 0
    normalized_matrix = np.nan_to_num(normalized_matrix)

    return normalized_matrix

def confusion_matrix(true_labels, predicted_labels,  normalize=False, visualize=False, kappa = None, path = None):
    """
    Calculate and optionally visualize the confusion matrix.
    
    Parameters:
        true_labels (numpy array): True labels.
        predicted_labels (numpy array): Predicted labels.
        normalize (bool): Flag to normalize the confusion matrix.
        visualize (bool): Flag to visualize the confusion matrix.
    
    Returns:
        numpy array: Confusion matrix.
    """
    # Get unique classes
    unique_classes = np.unique(np.concatenate((true_labels, predicted_labels)))
    num_classes = len(unique_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.int32)

    # Fill confusion matrix
    for i in range(len(true_labels)):
        matrix[true_labels[i], predicted_labels[i]] += 1
    
    # Normalize confusion matrix if required
    if normalize:
        matrix = normalize_confusion_matrix(matrix, axis=1)
        #matrix = np.round(matrix / len_query, 2)

    # Visualize confusion matrix if required
    if visualize:
        plt.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues,  vmin=0, vmax=10)
        plt.title(f"Confusion Matrix (Dataset 3, Cohens kappa: {round(kappa, 3)})")
        plt.colorbar()
       # tick_marks = np.arange(len(unique_classes))
      #  plt.xticks(tick_marks, unique_classes, rotation=45)
      #  plt.yticks(tick_marks, unique_classes)
        
        thresh = matrix.max() / 2.
        for i, j in [(i, j) for i in range(num_classes) for j in range(num_classes)]:
            plt.text(j, i, matrix[i, j], horizontalalignment="center", color="white" if matrix[i, j] > thresh else "black")
        
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
       # plt.tight_layout()

        if path:
            plt.savefig(path, dpi = 200)
        plt.show()

    return matrix

def plot_confusion_matrix(predictions,query_labels, N_QUERY, class_names=None, normalize=False, visualize=False, path=None):
    """
    Calculate and optionally visualize the confusion matrix.
    
    Parameters:
        loader (DataLoader): Data loader.
        model (torch.nn.Module): Trained model. 
        normalize (bool): Flag to normalize the confusion matrix.
        visualize (bool): Flag to visualize the confusion matrix.
        class_names (list of str): List of class names to label the axes.
        path (str): Optional path to save the confusion matrix plot.
    """


    # Get predicted labels from the model
    predicted_labels = predictions[:,-2].astype(int)

    # Step 1: Calculate Cohen's Kappa
    kappa = cohen_kappa_with_plot(query_labels, predicted_labels)
    print(f"Cohen's Kappa: {kappa}")
    print(query_labels.shape)

    # Step 2: Compute the confusion matrix
    cm = confusion_matrix(query_labels, predicted_labels)

    # Step 3: Normalize confusion matrix (if requested)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Step 4: Plot confusion matrix as heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm,vmin=0, vmax=N_QUERY, annot=True, fmt='.2f' if normalize else 'd', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'FSL (Datensatz 1) Confusion Matrix (Kappa: {kappa:.2f})')
    plt.tight_layout()
    # Step 5: Save the plot (if a path is provided)
    if path is not None:
        plt.savefig(path, dpi=200)

    # Step 6: Show the plot (if requested)
    if visualize:
        plt.show()

    # Return confusion matrix and kappa
    return cm, kappa



from sklearn.manifold import TSNE

def plot_dimension_reduction(loader, labels, model, class_list = None, path=None, method='tsne'):
    # Visualize embeddings using t-SNE


    query_features= loader[:,:-2]
    query_labels = labels
    prototypes = model.return_prototypes().cpu().detach().numpy()

    create_voronoi_diagram(query_features,  prototypes,query_labels, class_names = class_list ,method = method, path = path)

    
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from scipy.spatial import Voronoi
import matplotlib.colors as mcolors
from sklearn.cluster import KMeans

def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite Voronoi regions in a 2D diagram to finite regions.
    """
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def create_voronoi_diagram(points, prototypes, point_labels, class_names=None, method='tsne', path=None):
    """
    Create and plot a Voronoi diagram with each region in a different color,
    using TSNE or PCA to project points and prototypes into 2D space.

    Parameters:
    points (array-like): An array of shape (n, d) representing n points in d-dimensional space.
    prototypes (array-like): An array of shape (m, d) representing m prototype points in d-dimensional space.
    point_labels (array-like): An array of shape (n,) representing the labels of the points.
    class_names (list of str): Optional list of class names to use for labels.
    method (str): Dimensionality reduction method to use ('tsne' or 'pca').
    path (str): Optional path to save the plot.
    """
    import numpy as np
    all_data = np.vstack([points, prototypes])

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=0)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    all_data_2d = reducer.fit_transform(all_data)

    points_2d = all_data_2d[:len(points)]
    prototypes_2d = all_data_2d[len(points):]

    vor = Voronoi(prototypes_2d)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    fig, ax = plt.subplots(figsize=(10, 10))

    colormap = plt.cm.get_cmap('tab20')
    colors = colormap(np.linspace(0, 1, len(regions)))
    color_map = {i: colors[i] for i in range(len(regions))}

    patches = []

    for i, region in enumerate(regions):
        polygon = vertices[region]
        patches.append(Polygon(polygon, closed=True))

    p = PatchCollection(patches, facecolor='none', edgecolor='orange', alpha=0.4)

    # Set the colors for each patch
    p.set_facecolor([color_map[i] for i in range(len(patches))])
    ax.add_collection(p)

    # Predict labels based on closest prototype
    predicted_labels = np.argmin(np.linalg.norm(points[:, np.newaxis] - prototypes, axis=2), axis=1)

    # Identify correct and incorrect points
    correct_points = points_2d[point_labels == predicted_labels]
    incorrect_points = points_2d[point_labels != predicted_labels]

    # Plot points
    ax.plot(prototypes_2d[:, 0], prototypes_2d[:, 1], 'k*', markersize=10, label='Prototypes')  # Changed to black stars
    ax.plot(correct_points[:, 0], correct_points[:, 1], 'go', label='Correct Points')
    ax.plot(incorrect_points[:, 0], incorrect_points[:, 1], 'r^', label='Incorrect Points')

    # Create custom legend entries for Voronoi regions
    if class_names:
        for i, class_name in enumerate(class_names):
            if i in color_map:
                ax.plot([], [], color=color_map[i], label=class_name, linewidth=2.5)

    # Add legend outside of the plot area if necessary
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title='Class Names')

    # Adjust plot layout to accommodate legend

    ax.set_xlim([points_2d[:, 0].min() - 1, points_2d[:, 0].max() + 1])
    ax.set_ylim([points_2d[:, 1].min() - 1, points_2d[:, 1].max() + 1])

    ax.set_title(f'FSL Datensatz 1 Voronoi Diagram with Correct and Incorrect Points ({method.upper()})')
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.show()


def plot_dimension_reduction_kmeans(loader, model, class_list=None, path=None, device='cpu', method='tsne', n_clusters=10):
    # Visualize embeddings using t-SNE or PCA
    query_images, query_labels= next(iter(loader))
    
    with torch.no_grad():
        features = torch.nn.Sequential(*list(model.children())[:-1])(query_images.to(device))[:,:,0,0]
    labels = query_labels.cpu().numpy()

    create_voronoi_diagram_kmeans(features, labels, class_names=class_list, method=method, path=path, n_clusters=n_clusters)

def voronoi_finite_polygons_2d(vor, radius=None):
    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max() * 2

    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            new_regions.append(vertices)
            continue

        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                continue

            t = vor.points[p2] - vor.points[p1]
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def create_voronoi_diagram_kmeans(points, point_labels, class_names=None, method='tsne', path=None, n_clusters=10):
    """
    Create and plot a Voronoi diagram with each region in a different color,
    using TSNE or PCA to project points and k-means to find cluster centroids.

    Parameters:
    points (array-like): An array of shape (n, d) representing n points in d-dimensional space.
    point_labels (array-like): An array of shape (n,) representing the labels of the points.
    class_names (list of str): Optional list of class names to use for labels.
    method (str): Dimensionality reduction method to use ('tsne' or 'pca').
    path (str): Optional path to save the plot.
    n_clusters (int): Number of clusters for k-means.
    """

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(points)
    centroids = kmeans.cluster_centers_
    cluster_labels = kmeans.labels_




    # Assign the majority label of points in each cluster to the cluster
    cluster_majority_labels = np.zeros(n_clusters, dtype=int)
    for i in range(n_clusters):
        cluster_points_labels = point_labels[cluster_labels == i]
        if len(cluster_points_labels) > 0:
            cluster_majority_labels[i] = np.bincount(cluster_points_labels).argmax()

    all_data = np.vstack([points, centroids])

    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=0)
    elif method == 'pca':
        reducer = PCA(n_components=2)
    else:
        raise ValueError("Method must be 'tsne' or 'pca'")

    all_data_2d = reducer.fit_transform(all_data)

    points_2d = all_data_2d[:len(points)]
    centroids_2d = all_data_2d[len(points):]

    vor = Voronoi(centroids_2d)

    regions, vertices = voronoi_finite_polygons_2d(vor)

    fig, ax = plt.subplots(figsize=(10, 10))

    colormap = plt.cm.get_cmap('tab20')
    colors = colormap(np.linspace(0, 1, len(regions)))
    color_map = {i: colors[i] for i in range(len(regions))}

    patches = []

    for i, region in enumerate(regions):
        polygon = vertices[region]
        patches.append(Polygon(polygon, closed=True))

    p = PatchCollection(patches, facecolor='none', edgecolor='orange', alpha=0.4)

    # Set the colors for each patch
    p.set_facecolor([color_map[i] for i in range(len(patches))])
    ax.add_collection(p)

    # Predict labels based on majority label of the cluster
    predicted_labels = cluster_majority_labels[cluster_labels]

    # Identify correct and incorrect points
    correct_points = points_2d[point_labels == predicted_labels]
    incorrect_points = points_2d[point_labels != predicted_labels]

    # Plot points
    ax.plot(centroids_2d[:, 0], centroids_2d[:, 1], 'k*', markersize=10, label='Centroids')  # Changed to black stars
    ax.plot(correct_points[:, 0], correct_points[:, 1], 'go', label='Correct Points')
    ax.plot(incorrect_points[:, 0], incorrect_points[:, 1], 'r^', label='Incorrect Points')

    # Create custom legend entries for Voronoi regions
    if class_names:
        for i, class_name in enumerate(class_names):
            if i in color_map:
                ax.plot([], [], color=color_map[i], label=class_name, linewidth=2.5)

    # Add legend outside of the plot area if necessary
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small', title='Class Names')

    # Adjust plot layout to accommodate legend


   # ax.set_xlim([points_2d[:, 0].min() - 1, points_2d[:, 0].max() + 1])
   # ax.set_ylim([points_2d[:, 1].min() - 1, points_2d[:, 1].max() + 1])

    ax.set_title(f'Voronoi Diagram with Correct and Incorrect Points ({method.upper()})')
    plt.tight_layout()
    if path:
        plt.savefig(path, dpi=220, bbox_inches='tight') 
    plt.show()