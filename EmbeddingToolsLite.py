import os
import re
import cv2
import tensorflow as tf
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.cluster import AgglomerativeClustering, KMeans

tf_model_name = '/home/andrewhanigan/Desktop/repo_push/BRC-BIO/Embedding Models/TEST-BRC-BIO_Plastral-Pattern_Seg_RESNET101-BACKBONE_05-14-26_batchsize-8_epochs-14.keras'
autoencoder = tf.keras.models.load_model(tf_model_name)
autoencoder.summary()

#Since we trained our model as a full autoencoder, we can access the individual layer's ouptut like below
#This way we have the seperate encoder and decoder aspect of the model
encoder_input = autoencoder.input
encoder_output = autoencoder.get_layer('conv5_block3_out').output #Segmentation model currently uses max_pooling2d_4, ConvAE model uses conv2d_4, FCAE uses dense_3, resnet seg model uses conv5_block3_out
encoder_model = tf.keras.Model(encoder_input, encoder_output)
encoder_model.trainable = False
encoder_model.summary()
#BRC-BIO_Shell-And-Pattern_Seg_RESNET101-BACKBONE_05-14-26_batchsize-8_epochs-14.keras

#The n value here is the number of groups or clusters that Kmeans or the Agglomerative algorithms should find
n = 30

img_width = 64
img_height = 128
color_channel = 3

directory_to_test = "/home/andrewhanigan/Desktop/turtle_work/512x512 Turtle Images/August 24 Set/seg network/Test_Morgan/NORMALIZED/0"

def load_images_and_create_embeddings(directory):
    embeddings = []
    all_filenames = []

    list_of_images = os.listdir(directory)
    list_of_images.sort()
    for image_file in list_of_images:
        #img = Image.open(os.path.join(directory, image_file))
        #pil_img = img.resize((64,128), resample=Image.NEAREST)
        img = cv2.imread(os.path.join(directory, image_file))
        img = cv2.resize(img, (64,128), cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = np.array(img).astype('float32') / 255.
        tensor = tf.expand_dims(tf.convert_to_tensor(img), axis=0)
        embedding = encoder_model.predict(tensor)
        embeddings.append(embedding)
        all_filenames.append(image_file)
    return embeddings, all_filenames

all_embeddings, all_filenames = load_images_and_create_embeddings(directory_to_test)

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

def run_scikit_kmeans(list_of_all_embeddings, all_filenames_used, n, seed=43):
    """Set use_population to True if testing morgan and krause combined"""

    # Prepare embeddings
    embeddings_array = np.array(list_of_all_embeddings)
    embeddings_array = embeddings_array.reshape(embeddings_array.shape[0], -1)

    kmeans = KMeans(n_clusters=n,random_state=seed)
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
                krause_count += 1
                clutch_labels.append(1)
            else:
                morgan_count += 1
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
    return score

#run_scikit_agglom_algo(list_of_all_embeddings=all_embeddings, all_filenames_used=all_filenames, n=n)
#run_scikit_kmeans(list_of_all_embeddings=all_embeddings, all_filenames_used=all_filenames, n=n)

def calc_kmeans_mean_stdv():
    list_of_aris = []
    for _ in range(1000):
        ari = run_scikit_kmeans(list_of_all_embeddings=all_embeddings, all_filenames_used=all_filenames, n=n, seed=None)
        list_of_aris.append(ari)
        print(_)
    mean = np.mean(list_of_aris)
    print(f"Final mean value: {mean}")
    std = np.std(list_of_aris)
    print(f"Standard Deviation: {std}" )

calc_kmeans_mean_stdv()