"""This script contains the tools we can use on the embeddings created from our
Autoencoder"""

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import copy
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
import umap
import pandas as pd
import math
from statistics import median
from PIL import Image
from scipy.cluster.hierarchy import dendrogram
import tkinter as tk
import shared
import re


n = 18

#Forces tf to run on the CPU
#tf.config.set_visible_devices([], 'GPU')

img_width = 64
img_height = 128
color_channel = 3
seg_model = False

model_name = 'brc_bio_conv-network_fs=8x4x64.keras'
autoencoder = tf.keras.models.load_model(model_name)

autoencoder.summary()


#Since we trained our model as a full autoencoder, we can access the individual layer's ouptut like below
#This way we have the seperate encoder and decoder aspect of the model
encoder_input = autoencoder.input
encoder_output = autoencoder.get_layer('conv2d_4').output #Segmentation model currently uses max_pooling2d_4, ConvAE model uses conv2d_4, FCAE uses dense_3, resnet seg model uses conv5_block3_out
encoder_model = tf.keras.Model(encoder_input, encoder_output)
encoder_model.trainable = False
encoder_model.summary()

directory_to_test = '/home/andrewhanigan/Desktop/turtle_work/512x512 Turtle Images/August 24 Set/seg network/Test_Krause/NORMALIZED/'

def __add_random_noise_to_images(image):
    noise_factor = 0.1
    
    x_test_noisy = image + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image[0].shape) 
    x_test_noisy = np.clip(x_test_noisy, 0., 1.)

    return x_test_noisy

def load_and_preprocess_data(add_noise=False, directory='', segmentation_model=False):
    class_folders  = os.listdir(directory)
    class_folders.sort()
    print(class_folders)

    images_to_test = []
    decoded_imgs = []
    embeddings = []
    embedding_class = []
    all_filenames_used = []
    decoder_model_test = []
    list_of_embedding_objs = []
    for folder in class_folders:
        test_image_dir = os.path.join(directory, folder)
        test_image_filenames = [file for file in os.listdir(test_image_dir) if file.endswith('.png') or file.endswith('.jpg')]
        test_image_filenames.sort()
        index = 0
        for file in test_image_filenames:
            img = cv2.imread(os.path.join(test_image_dir, file))
            img = cv2.resize(img, (64,128), cv2.INTER_NEAREST)
            pil_img = Image.open(os.path.join(test_image_dir, file))
            if color_channel == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #to convert to grayscale
            if color_channel == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #cv2 loads images in BGR format, this line switches the img  to RGB
            img  = np.array(img).astype('float32') / 255.
            tensor = tf.expand_dims(tf.convert_to_tensor(img), axis=0) 
            print(img.shape)
            if add_noise == True:
                img = __add_random_noise_to_images(img)
            images_to_test.append(img)
            
            decoded_imgs.append(autoencoder.predict(tensor))
            embedding = encoder_model.predict(tensor)
            embeddings.append(embedding)
            all_filenames_used.append(file)
            if segmentation_model:
                pil_img = pil_img.resize((64,128), resample=Image.BICUBIC)
                image = tf.convert_to_tensor(pil_img)
                image = tf.cast(image, tf.float32)/255.0
                modeled = tf.math.argmax(autoencoder.predict(image[tf.newaxis, ...]),axis=-1)
                modeled = modeled[...,tf.newaxis]
                modeled = modeled[0]
                modeled = modeled * 20
                predicted_segmentation_image = keras.utils.array_to_img(modeled)
                
                #Stores the class value asscociated with the embedding
                embedding_class.append(int(folder))     
            else:
                #Stores the class value asscociated with the embedding
                embedding_class.append(int(folder))


    
    return images_to_test, decoded_imgs, embeddings, embedding_class, all_filenames_used, list_of_embedding_objs


images_to_test, decoded_imgs, embeddings, embedding_class, all_filenames_used, list_of_embedding_objs = load_and_preprocess_data(add_noise=False, directory=directory_to_test, segmentation_model=seg_model)
#np.save('embedding_objs.npy', list_of_embedding_objs)

def __display_image_and_decoded(images_to_test, decoded_imgs):
    number_of_images = 7
    plt.figure(figsize=(20, 6))
    for i in range(1, number_of_images + 1):
        # Display original, before encoding
        ax = plt.subplot(3, number_of_images, i)
        plt.imshow(images_to_test[i].reshape(img_height, img_width, color_channel))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction, after decoding
        ax = plt.subplot(3, number_of_images, i + number_of_images)
        plt.imshow(decoded_imgs[i].reshape(img_height, img_width, color_channel))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        """ # Display reconstruction, after decoding
        ax = plt.subplot(3, number_of_images, i + 2 * number_of_images)
        plt.imshow(decoder_model_test[i].reshape(img_height, img_width, color_channel))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False) """
    plt.show()

#__display_image_and_decoded(images_to_test, decoded_imgs)

def apply_tsne(embeddings, embedding_class, filenames):
    flattened_tensors = []
    for tensor in embeddings:
        tensor = tensor.reshape( -1)
        flattened_tensors.append(tensor)
    #reshaped_encoded_imgs = encoded_imgs.reshape(num_images, -1)  # Flatten the feature dimensions 

    tsne = TSNE(n_components=2, perplexity=8, init='pca')
    tsne_features = tsne.fit_transform(np.array(flattened_tensors))
    embedding_class = np.array(embedding_class)

    # Plot the TSNE results without label
    plt.figure(figsize=(10, 6))
    plt.scatter(tsne_features[:, 0], tsne_features[:, 1], c=embedding_class, cmap='Dark2') #, cmap='Dark2'
    for i, txt in enumerate(filenames):
        plt.annotate(txt, (tsne_features[i, 0], tsne_features[i, 1]), fontsize=8)

    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.colorbar()
    plt.show()

#apply_tsne(embeddings, embedding_class, all_filenames_used)

def apply_umap(embeddings, embedding_class, image_files):
        # Initialize UMAP
    umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)

    # Fit and transform the embeddings
    embeddings = tf.convert_to_tensor(embeddings)
    print(type(embeddings))
    umap_embeddings = umap_reducer.fit_transform(tf.reshape(embeddings, (embeddings.shape[0], -1)))
    embedding_image_map = {image_files[i]: umap_embeddings[i] for i in range(len(image_files))}

    # Plot the results
    plt.scatter(umap_embeddings[:, 0], umap_embeddings[:, 1], c=embedding_class, cmap='Dark2')
    for i, txt in enumerate(image_files):
        plt.annotate(txt, (umap_embeddings[i, 0], umap_embeddings[i, 1]), fontsize=8)
    plt.colorbar()
    plt.title('UMAP projection of the embeddings')
    plt.show()

#apply_umap(embeddings, embedding_class, all_filenames_used)

def euclidean_dist_compare(list_of_all_embeddings, file_names, embedding_class):
    def flatten(embedding):
        return embedding.flatten()

    # Separate embeddings
    query_embeddings = [flatten(e) for i, e in enumerate(list_of_all_embeddings) if embedding_class[i] == 1]
    other_embeddings = [flatten(e) for i, e in enumerate(list_of_all_embeddings) if embedding_class[i] == 0]

    # Just take the first from class 0 as the query
    query = query_embeddings[0]

    # Compute Euclidean distances
    distances = [np.linalg.norm(query - other) for other in other_embeddings]

    # Find the index of the closest embedding
    closest_index = np.argmin(distances)
    closest_distance = distances[closest_index]

    print(f"Closest embedding index (among class 1): {closest_index}")
    print(file_names[closest_index])
    print(f"Distance to query: {closest_distance}")

#euclidean_dist_compare(list_of_all_embeddings=embeddings, file_names=all_filenames_used, embedding_class=embedding_class)


def three_dimensional_visualization(embeddings, embedding_class):
    """Currently uses PCA on the embeddings for plotting the embeddings"""
    flattened_tensors = []
    for tensor in embeddings:
        tensor = tensor.reshape( -1)
        flattened_tensors.append(tensor)
    pca = PCA(n_components=3)
    reduced_embeddings = pca.fit_transform(flattened_tensors)

    # Create a scatter plot
    fig = go.Figure(data=go.Scatter3d(
        x=reduced_embeddings[:,0],
        y=reduced_embeddings[:,1],
        z=reduced_embeddings[:,2],
        mode='markers',
        marker=dict(
            size=5,
            color=embedding_class,  
            colorscale='Viridis',  
            opacity=0.8,
            colorbar=dict(title='Class')
        )
    ))
    fig.update_layout(
        title='3D Visualization of Embeddings',
        scene=dict(
            xaxis_title='PC1',
            yaxis_title='PC2',
            zaxis_title='PC3'
        )
    )

    fig.show()
#three_dimensional_visualization(embeddings, embedding_class)


def calculate_and_display_cdist(test_images_dir):
    """Compares embeddings using the cdist formula"""
    
    def load_test_data(test_images_dir):
        test_files = [file for file in os.listdir(test_images_dir) if file.endswith('.jpg') or file.endswith('.jpeg')]
        test_images = []
        for file in test_files:
            img = cv2.imread(os.path.join(test_images_dir, file))
            if color_channel == 1:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if color_channel == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            test_images.append(img)
        x_test = np.array(test_images).astype('float32') / 255.
        x_test = np.reshape(x_test, (len(x_test), 128, 64, color_channel))

        return x_test, test_files
    
    x_test, file_names = load_test_data(test_images_dir)
    encoded_imgs = encoder_model.predict(x_test)

    encoded_imgs_flatten = encoded_imgs.reshape((len(x_test), np.prod(encoded_imgs.shape[1:])))
    np.random.seed(0)
    selected_indices = np.random.choice(x_test.shape[0], 5, replace=False)

    def find_similar_images(embeddings, selected_indices):
        """Find and return indices of similar images based on embeddings."""
        similar_images_indices = []
        similar_images_distances = []
        for index in selected_indices:
            distances = cdist(embeddings[index:index+1], embeddings, 'euclidean')
            closest_indices = np.argsort(distances)[0][1:5] # The closest euclidean distance calculated would be 0 from comparing the embedding against the embedding, so we exclude the first element
            similar_images_indices.append(closest_indices)
            distances_to_add =[]
            for index in closest_indices:
                distances_to_add.append(distances[0][index])
            similar_images_distances.append(distances_to_add)


        return similar_images_indices, similar_images_distances

    def display_similar_images(x_test, selected_indices, similar_images_indices, similar_images_distances, file_names):
        """Visualize the original and similar images."""
        plt.figure(figsize=(10, 7))
        for i, (original_image_index, similar_indices, euclid_dist) in enumerate(zip(selected_indices, similar_images_indices, similar_images_distances)):
            ax = plt.subplot(5, 5, i * 5 + 1)
            plt.imshow(x_test[original_image_index].reshape(128, 64, color_channel))
            plt.title(f"{file_names[original_image_index]}", fontsize=7)
            plt.gray()
            ax.axis('off')
            
            for j, sim_index in enumerate(similar_indices):
                ax = plt.subplot(5, 5, i * 5 + j + 2)
                plt.imshow(x_test[sim_index].reshape(128, 64, color_channel))
                plt.title(f"{file_names[similar_indices[j]]}", fontsize=5)
                plt.gray()
                plt.text(0.5, -0.1, f"{euclid_dist[j]:.5f}", ha='center', va='top', transform=ax.transAxes)
                ax.axis('off')

        plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1, wspace=0.5, hspace=0.5)
        plt.show()

    similar_images_indices, similar_images_distances = find_similar_images(encoded_imgs_flatten, selected_indices)
    display_similar_images(x_test, selected_indices, similar_images_indices, similar_images_distances, file_names)

#calculate_and_display_cdist(test_images_dir='/home/andrewhanigan/Desktop/512x512 Turtle Images/August 24 Set/Clutch Testing Set/All')
    
def visualize_embedding_features(embeddings, filenames):
    index = 0
    print(len(embeddings))
    for index in range(len(embeddings)):

        fig, ax = plt.subplots()
        
        embedding = np.array(embeddings[index][0])
        grid = np.zeros((4 * 4, 2 * 4))

        # Fill the grid with the normalized embedding values
        for i in range(16):
            for j in range(8):
                # Reshape the 16-dimensional vector into a 4x4 grid
                grid_4x4 = embedding[i, j].reshape(4, 4)
                # Place the 4x4 grid into the larger grid
                grid[i*4:(i+1)*4, j*4:(j+1)*4] = grid_4x4

        cax = ax.imshow(grid, cmap='viridis', vmin=0, vmax=0.4)
        ax.set_title(filenames[index])

        # Plot the grid
        fig.colorbar(cax)
        plt.show()

#visualize_embedding_features(embeddings, all_filenames_used)
        
def calc_vector_distance(embeddingA, embeddingB):

    max_distances = []
    median_distances = []
    vector_distances = []
    avg_distances = []
    temp = []
    for i in range(len(embeddingA[0])):
        for j in range(len(embeddingA[0][0])):
            for m in range(len(embeddingA[0][0][0])):
                distance_value = math.sqrt(((embeddingA[0][i][j][m] - embeddingB[0][i][j][m]) ** 2))
                temp.append(distance_value)
        
                vector_distances.append(distance_value)
            max_distances.append(max(temp))
            median_distances.append(median(temp))
            temp = []
    
    #print(vector_distances)
    return vector_distances, max_distances, median_distances




def write_vector_distances(target_embedding, embeddings_to_compare, all_filenames_used, avg=True, median=True, max=True):
    print("Target embedding: " + all_filenames_used[0])
    print(target_embedding.shape)
    vector_euclid_distances = []
    for i in range(len(embeddings_to_compare) - 1):
        print('Comparing against: ' + all_filenames_used[i])
        vector_distances, max_distances, median_distances = calc_vector_distance(target_embedding, embeddings_to_compare[i])
        index = 0
        for j in range(len(target_embedding[0])):
            for m in range(len(target_embedding[0][0])):
                print(max_distances[index], end=' ')
                index += 1
            print('\n')
        print('--------\n')
    

#write_vector_distances(embeddings[0], embeddings, all_filenames_used)

     
def write_embeddings_to_csv(embeddings, filenames):

    flattened_embeddings = [embedding.flatten() for embedding in embeddings]
    df = pd.DataFrame(flattened_embeddings)
    df_transposed = df.T
    df_transposed.loc['Filename'] = filenames
    df_transposed.to_excel('brc_bio_seg_network.xlsx', index=False, header=False)

    
#write_embeddings_to_csv(embeddings, all_filenames_used)

def is_proper_subset(listA, listB):
    flat_a = np.array(listA).flatten()
    flat_b = np.array(listB).flatten()
    set_a = set(flat_a)
    set_b = set(flat_b)
    
    return set_a.issubset(set_b)  

def get_minimum_and_maximum_values_at_each_index_of_all_embeddings(list_of_embeddings):
    min_values =[]
    max_values = []
    list_of_indices_to_delete = []
    index  = 0
    for index in range(len(list_of_embeddings[0].flattened_data)):
        min_val = list_of_embeddings[0].flattened_data[index]
        max_val = list_of_embeddings[0].flattened_data[index]
        for embedding in list_of_embeddings:
            if embedding.flattened_data[index] < min_val:
                min_val = embedding.flattened_data[index]
            if embedding.flattened_data[index] > max_val:
                max_val = embedding.flattened_data[index]
        if max_val != 0 and max_val != min_val:
            min_values.append(min_val)
            max_values.append(max_val)
        elif max_val == min_val:
            list_of_indices_to_delete.append(index)
        else:
            list_of_indices_to_delete.append(index)
    for embedding in list_of_embeddings:
        embedding.flattened_data = np.delete(embedding.flattened_data, list_of_indices_to_delete)
    return min_values, max_values

def normalize_and_update_embedding_data(list_of_embeddings, min_values, max_values):
    list_of_updated_embeddings = []
    index = 0
    for embedding in list_of_embeddings:
        feature_space = embedding.flattened_data.copy()
        new_fspace = []
        new_fspace.clear()
        for index in range(len(feature_space)):
            updated_value = ((feature_space[index] - min_values[index]) / (max_values[index] - min_values[index])) * 100
            new_fspace.append(updated_value)
        embedding_data = np.array(new_fspace.copy())
        new_embedding = AgglomerativeAlgo.Embedding(embedding.filename, embedding.image, embedding.predicted_segmentation_image, embedding_data)
        list_of_updated_embeddings.append(new_embedding)
    return list_of_updated_embeddings

def update_cluster_id():
    shared.cluster_id += 1

def gen_clutch_and_cluster_labels(iteration):
    """Genreates both the predicted and ground truth labels for which clucth or cluster
    the embedding belnogs to

    """
    ground_truth_clutch_labels = []
    cluster_result_labels = []

    for clustering in iteration.list_of_clusterings_in_iteration:
        for cluster in clustering.list_of_clusters:
            for embedding in cluster.list_of_embeddings:
                ground_truth_clutch_labels.append(embedding.clutch_id)
                cluster_result_labels.append(cluster.cluster_id)
    
    return ground_truth_clutch_labels, cluster_result_labels

def initialize_first_iteration_of_clusterings(embeddings_to_use):
    """Creates the first iteration of clustering sets for a group of embeddings"""
    initial_distance = 0
    list_of_clusters_for_clustering = []
    list_of_embeddings = [] #used to force singular embeddings in a cluster to match list datatype for later clusters, all clusters at this point will contain one embedding
    for embedding in embeddings_to_use:
        fspace = embedding.flattened_data
        vectors_from_fspace = embedding.create_initial_vectors_from_feature_space(fspace, initial_distance)
        list_of_embeddings.append(embedding)
        new_cluster = AgglomerativeAlgo.ClusterDescription(number_of_embeddings=1, list_of_vectors=vectors_from_fspace.copy(), list_of_embeddings=list_of_embeddings.copy(), cluster_id=shared.cluster_id)      
        update_cluster_id()
        list_of_clusters_for_clustering.append(new_cluster)
        vectors_from_fspace.clear()
        list_of_embeddings.clear()
    new_clustering = AgglomerativeAlgo.Clustering(list_of_clusters=list_of_clusters_for_clustering)
    clustering_for_iteration = []
    clustering_for_iteration.append(new_clustering)

    first_iteration = AgglomerativeAlgo.Iteration(list_of_clusterings_in_iteration=clustering_for_iteration.copy(), iteration=0)

    clutch_labels, cluster_labels = gen_clutch_and_cluster_labels(first_iteration)
    first_iteration.rand_score = adjusted_rand_score(clutch_labels, cluster_labels)

    return first_iteration

def create_clusterings(list_of_all_embeddings, all_filenames_used):
    """Takes a list of embeddings and creates clusterings 
    at each (x,y) position of an (x,y,z) embedding.

    Assumes all embeddings in list_of_all_embeddings are of the same shape
    """
    total_count_of_embeddings = len(list_of_all_embeddings)
    print('Creating clusterings for: ' + str(total_count_of_embeddings) +' embeddings')
    list_of_all_embeddings[0].print_embedding_dimensions()
    embeddings_to_use = list_of_all_embeddings.copy()
    min_values, max_values = get_minimum_and_maximum_values_at_each_index_of_all_embeddings(embeddings_to_use)
    list_of_updated_embeddings = normalize_and_update_embedding_data(embeddings_to_use, min_values, max_values)
    list_of_all_iterations = []
    first_iteration = initialize_first_iteration_of_clusterings(list_of_updated_embeddings)
    list_of_all_iterations.append(first_iteration)
    iteration_index = 1
    merge_history = [] #Used for dendogram
    #Gen the iterations
    while list_of_all_iterations[0].list_of_clusterings_in_iteration[0].number_of_clusters != 1:
        clustering = copy.deepcopy(list_of_all_iterations[0].list_of_clusterings_in_iteration[0]) #Creating a copy of past clustering to not mess with the values of the previous iterations
        cluster_to_mergeA = clustering.nearest_cluster_descriptions[0]
        cluster_to_mergeB = clustering.nearest_cluster_descriptions[1]
        clustering.list_of_clusters.remove(cluster_to_mergeA) #Removing the two merged cluster descriptions from the set of all cluster descriptions
        clustering.list_of_clusters.remove(cluster_to_mergeB)
        merged_cluster = cluster_to_mergeA.merge_cluster_descriptions(cluster_to_mergeB)
        merge_data = [float(cluster_to_mergeA.cluster_id), float(cluster_to_mergeB.cluster_id), float(clustering.clustering_distance),float(merged_cluster.number_of_embeddings)]
        merge_history.append(merge_data)
        for cluster in clustering.list_of_clusters:
            if is_proper_subset(cluster.list_of_vectors, merged_cluster.list_of_vectors):
                print('Subset found')
                merged_cluster = merged_cluster.merge_cluster_descriptions(cluster) #Merging the subset with the overlapping cluster hyper-rectangle
                clustering.list_of_clusters.remove(cluster) #If a subset is detected and merged, it is also removed from the list of remaining embeddings
        list_of_clusters_for_clustering = []
        for cluster in clustering.list_of_clusters:
            list_of_clusters_for_clustering.append(cluster)
        list_of_clusters_for_clustering.append(merged_cluster) #Adding the newly merged cluster descirption to the list of clusterings
        new_clustering = AgglomerativeAlgo.Clustering(list_of_clusters_for_clustering.copy())
        clustering_for_iteration = []
        clustering_for_iteration.append(new_clustering)
        new_iteration = AgglomerativeAlgo.Iteration(list_of_clusterings_in_iteration=clustering_for_iteration.copy(), iteration=iteration_index)

        clutch_labels, cluster_labels = gen_clutch_and_cluster_labels(new_iteration)
        
        new_iteration.rand_score = adjusted_rand_score(clutch_labels, cluster_labels)
        print(new_iteration.rand_score)
        
        iteration_index += 1
        list_of_all_iterations.insert(0, new_iteration)

    #The following part of this function displays info on the iterations generated    
    best_iteration = copy.deepcopy(list_of_all_iterations[0])
    second_best_iteration = copy.deepcopy(list_of_all_iterations[0])
    iteration_with_best_rand = copy.deepcopy(list_of_all_iterations[0])
    for iteration in list_of_all_iterations:
        if iteration.list_of_clusterings_in_iteration[0].goodness >= best_iteration.list_of_clusterings_in_iteration[0].goodness:
            second_best_iteration = copy.deepcopy(best_iteration)
            best_iteration = copy.deepcopy(iteration)
        if iteration.rand_score > iteration_with_best_rand.rand_score:
            iteration_with_best_rand = iteration
    print('\n*Best iteration: *\n')
    print(best_iteration)

    

    print('\n*Second best iteration*\n')
    print(second_best_iteration)

    print('\n*Iteration with best Rand Score: *\n')
    print(iteration_with_best_rand)

    #Gui to view iteration's clusterings
    root = tk.Tk()
    view = AgglomerativeAlgo.IterationClusterViewer(root, iteration_with_best_rand)
    root.mainloop()

    #The tree diagram
    linkage_matrix = np.array(merge_history)
    plt.figure(figsize=(10, 5))
    dendrogram(linkage_matrix, labels=all_filenames_used, leaf_rotation=90, leaf_font_size=8)
    plt.title("Dendrogram")
    plt.xlabel("Images")
    plt.ylabel("Distance")
    plt.show()

#create_clusterings(list_of_embedding_objs, all_filenames_used)


def run_scikit_agglom_algo(list_of_all_embeddings, all_filenames_used, n):
    """Set use_population to True if testing morgan and krause combined"""

    # Prepare embeddings
    embeddings_array = np.array(list_of_all_embeddings)
    embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)

    # Run clustering
    agg_clustering = AgglomerativeClustering(
        n_clusters=n,
        metric='euclidean',
        linkage='ward'
    )

    labels = agg_clustering.fit_predict(embeddings_array)

    print("Cluster labels:", labels)
    clusters = {}
    clutch_labels = []
    cluster_labels = list(labels)

    morgan_count = 0
    krause_count = 0

    for i, label in enumerate(labels):
        filename = all_filenames_used[i]

        # Group filenames by cluster
        clusters.setdefault(label, []).append(filename)

        if n != 2:
            # Population mode (krause and morgan combined)
            if "krause" in filename:
                krause_count += 1
                match = re.search(r'clutch20(\d+)', filename)
                if match:
                    clutch_labels.append(int(match.group(1)) * 10)
                else:
                    print("No clutch number found:", filename)

            else:
                morgan_count += 1
                match = re.search(r'clutch(\d+)', filename)
                if match:
                    clutch_labels.append(int(match.group(1)))
                else:
                    print("No clutch number found:", filename)

        if n == 2:
            if "krause" in filename:
                clutch_labels.append(1)
            else:
                clutch_labels.append(2)
    print(f"\nTotal samples: {len(labels)}")
    print(f"Clutch labels: {len(clutch_labels)}")
    print(f"Cluster labels: {len(cluster_labels)}")

    if len(clutch_labels) != len(cluster_labels):
        raise ValueError("Mismatch between clutch_labels and cluster_labels lengths!")

    print(f"Morgan count: {morgan_count}")
    print(f"Krause count: {krause_count}")

    # -------------------------
    # Print clusters
    # -------------------------
    for cluster_id, filenames in clusters.items():
        print(f"Cluster {cluster_id}: {filenames}\n")

    # -------------------------
    # Score
    # -------------------------
    print("***************")
    print("Agglomerative score:")
    score = adjusted_rand_score(clutch_labels, cluster_labels)
    print(score)

def run_scikit_kmeans(list_of_all_embeddings, all_filenames_used, n):
    """Set use_population to True if testing morgan and krause combined"""

    # Prepare embeddings
    embeddings_array = np.array(list_of_all_embeddings)
    embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)

    kmeans = KMeans(n_clusters=n,random_state=43)
    labels = kmeans.fit_predict(np.array(embeddings_array))

    print("Cluster labels:", labels)
    clusters = {}
    clutch_labels = []
    cluster_labels = list(labels)

    morgan_count = 0
    krause_count = 0

    for i, label in enumerate(labels):
        filename = all_filenames_used[i]

        # Group filenames by cluster
        clusters.setdefault(label, []).append(filename)

        if n != 2:
            # Population mode (krause and morgan combined)
            if "krause" in filename:
                krause_count += 1
                match = re.search(r'clutch20(\d+)', filename)
                if match:
                    clutch_labels.append(int(match.group(1)) * 10)
                else:
                    print("No clutch number found:", filename)

            else:
                morgan_count += 1
                match = re.search(r'clutch(\d+)', filename)
                if match:
                    clutch_labels.append(int(match.group(1)))
                else:
                    print("No clutch number found:", filename)

        if n == 2:
            if "krause" in filename:
                clutch_labels.append(1)
            else:
                clutch_labels.append(2)
    print(f"\nTotal samples: {len(labels)}")
    print(f"Clutch labels: {len(clutch_labels)}")
    print(f"Cluster labels: {len(cluster_labels)}")

    if len(clutch_labels) != len(cluster_labels):
        raise ValueError("Mismatch between clutch_labels and cluster_labels lengths!")

    print(f"Morgan count: {morgan_count}")
    print(f"Krause count: {krause_count}")

    # -------------------------
    # Print clusters
    # -------------------------
    for cluster_id, filenames in clusters.items():
        print(f"Cluster {cluster_id}: {filenames}\n")

    # -------------------------
    # Score
    # -------------------------
    print("***************")
    print("Kmeans score:")
    score = adjusted_rand_score(clutch_labels, cluster_labels)
    print(score)


#Example of how to run clustering algorithms
#run_scikit_agglom_algo(list_of_all_embeddings=embeddings, all_filenames_used=all_filenames_used, n=n)
#run_scikit_kmeans(list_of_all_embeddings=embeddings, all_filenames_used=all_filenames_used, n=n)
