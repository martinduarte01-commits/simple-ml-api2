

FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --no-cache-dir \
    pandas \
    numpy \
    scikit-learn \
    matplotlib

RUN mkdir -p models

RUN python model.py

import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

# Setup Seaborn
sns.set_style("whitegrid")
sns.set_context("talk")

orig_url = "https://docs.google.com/spreadsheets/d/1mlXah6ADb88FI6bL0pIQap0gxD7IoimT/edit?usp=sharing"
file_id = orig_url.split('/')[-2]
data_path='https://drive.google.com/uc?export=download&id=' + file_id
df_offers = pd.read_excel(data_path, sheet_name=0)

# subset data to get useful features
df_offers = df_offers.iloc[:, :6]
df_offers.columns = ["offer_id", "campaign", "wine_variety", "min_qty", "discount", "origin"]
df_offers.head(15)

"""The second dataset below in the second worksheet contains transactional data i.e., which offer each customer responded to and ended up buying it"""

orig_url = "https://docs.google.com/spreadsheets/d/1mlXah6ADb88FI6bL0pIQap0gxD7IoimT/edit?usp=sharing"
file_id = orig_url.split('/')[-2]
data_path='https://drive.google.com/uc?export=download&id=' + file_id
df_transactions = pd.read_excel(data_path, sheet_name=1)

df_transactions.columns = ["customer_name", "offer_id"]
df_transactions['frequency'] = 1
df_transactions.head(15)

df_merged = df_offers.merge(df_transactions, how='left', on='offer_id')
df_merged.head(20)

df_merged.shape


X = pivot.values
X

X.shape



from sklearn.cluster import KMeans

ss = []
cluster_ranges = range(2, 11)

for K in cluster_ranges:
    # fit a k-means model
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=1000, max_iter=10000)
    kmeans.fit_predict(X)
    # stores the ss errors.
    ss.append(kmeans.inertia_)

ax = sns.lineplot(x=cluster_ranges, y=ss)
ax.set_title("Error vs K for Wine Offer Data")
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('SS Error');



from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
import matplotlib.cm as cm
import numpy as np

avg_silhouette_scores = []
cluster_ranges = range(2, 7)

# Get rid of warning of too many plots open
plt.rcParams.update({'figure.max_open_warning': 0})

for K in cluster_ranges:
    # Create a subplot with 1 row and 2 columns
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    # The 1st subplot is the silhouette plot
    # The silhouette coefficient can range from -1, 1
    ax.set_xlim([-.4, 1])
    # The (K+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax.set_ylim([0, len(X) + (K + 1) * 10])

    # Initialize the clusterer with n_clusters value and a random generator
    # seed of 42 for reproducibility.
    kmeans_model = KMeans(n_clusters=K, random_state=42, n_init=1000, max_iter=10000)
    cluster_labels = kmeans_model.fit_predict(X)

    # The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = (silhouette_score(X, cluster_labels))
    avg_silhouette_scores.append(silhouette_avg)

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    y_lower = 10
    for i in range(K):
        # Aggregate the silhouette scores for samples belonging to
        # cluster i, and sort them
        ith_cluster_silhouette_values = \
            sample_silhouette_values[cluster_labels == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / K)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title(f"Silhouette analysis for KMeans clustering with n_clusters = {K}")
    ax1.set_xlabel("Silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # 2nd Plot showing the actual clusters formed
    colors = cm.nipy_spectral((cluster_labels.astype(float) + 1) / K)
    # to visualize the actual data in a continuous space we use PCA
    pca = PCA(n_components=2).fit(X)
    data2D = pca.transform(X)
    ax2.scatter(data2D[:, 0], data2D[:, 1], marker='o', s=30, lw=0, alpha=0.7,
                c=colors, edgecolor='k')

    # Labeling the clusters
    centers = kmeans_model.cluster_centers_
    # Draw white circles at cluster centers


    # Draw white circles at cluster centers
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                c="white", alpha=1, s=200, edgecolor='k')

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                    s=50, edgecolor='k')


    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel(f"Feature space for the 1st feature : {pivot.columns[0][1]}")
    ax2.set_ylabel(f"Feature space for the 2nd feature : {pivot.columns[1][1]}")

    plt.suptitle(
        (
            f"Silhouette analysis for KMeans clustering on sample data with K = {K}"
        ),
        fontsize=14,
        fontweight="bold",
    )
print("\n")

plt.show()

"""Now plot average Silhouette Score for each value of K."""

ax = sns.lineplot(x=cluster_ranges, y=avg_silhouette_scores)
ax.set_title('K vs Average Silhouette Score')
ax.set_xlabel('Number of Clusters')
ax.set_ylabel('Average Silhouette Score');

avg_silhouette_scores

from sklearn.decomposition import PCA

# First calculate top two PCA features.
pca = PCA(n_components=2)
pca_features = pca.fit_transform(X)
x = pca_features[:,0]
y = pca_features[:,1]

K=3

kmeans = KMeans(n_clusters=K, random_state=42, n_init=1000, max_iter=10000)
cl_labels = kmeans.fit_predict(X)
df_pca_features = pd.DataFrame({'cluster_id': cl_labels,
                                'pc_1':x,
                                'pc_2':y})
df_pca_features

ax = sns.scatterplot(x='pc_1', y='pc_2',
                hue='cluster_id',
                size='cluster_id',
                data=df_pca_features,
                palette='tab20',
                legend='full')
ax.set_xlabel('PCA Feature 1')
ax.set_ylabel('PCA Feature 2')
ax.set_title('PCA results of KMeans with ' + str(K) + ' clusters')
ax.legend(bbox_to_anchor=(1.04,1.02), loc='upper left', fontsize='large');


cl_labels

customer_segments = pd.DataFrame({
    'customer_name': pivot.index.tolist(),
    'Cluster': cl_labels
})

customer_segments.head(10)

df_merged.head(10)

df_cluster_merged = df_merged.merge(customer_segments, on='customer_name')
df_cluster_merged.head(10)


months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August',
                                           'September', 'October', 'November', 'December']
res = pd.crosstab(df_cluster_merged.Cluster, df_cluster_merged.campaign)
res = res[months]
res

from matplotlib import cm

custom_colors = cm.get_cmap('tab20', 12)

fig, ax = plt.subplots(figsize=(8, 6))

res.plot(kind='bar', ax=ax, cmap=custom_colors)
ax.legend(bbox_to_anchor=(1.02, 1.02), fontsize='large')
plt.xlabel('Customer Segment')
plt.xticks(rotation=0)
plt.ylabel('Wine Offers Purchased')
plt.title('Wine Offers Purchased by Customer Segment');


res[res <= 15] = 0
res

fig, ax = plt.subplots(figsize=(8, 6))

res.plot(kind='bar', ax=ax, cmap=custom_colors)
ax.legend(bbox_to_anchor=(1.02, 1.02), fontsize='large')
plt.xlabel('Customer Segment')
plt.xticks(rotation=0)
plt.ylabel('Wine Offers Purchased')
plt.title('Wine Offers Purchased by Customer Segment > 15 Offers');



res = pd.crosstab(df_cluster_merged.Cluster, df_cluster_merged.wine_variety)
res

fig, ax = plt.subplots(figsize=(8, 6))

res.plot(kind='bar', ax=ax, cmap=custom_colors)
ax.legend(bbox_to_anchor=(1.02, 1.02), fontsize='large')
plt.xlabel('Customer Segment')
plt.xticks(rotation=0)
plt.ylabel('Wine Type Purchased')
plt.title('Wine Type Purchased by Customer Segment');



res = pd.crosstab(df_cluster_merged.Cluster, df_cluster_merged.min_qty)
res

fig, ax = plt.subplots(figsize=(8, 6))

res.plot(kind='bar', ax=ax)
ax.legend(bbox_to_anchor=(1.02, 1.02), fontsize='large')
plt.xlabel('Customer Segment')
plt.xticks(rotation=0)
plt.ylabel('Wine Quantity Purchased')
plt.title('Wine Quantity Purchased by Customer Segment');

