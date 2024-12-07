import torch
import numpy as np
from PIL import Image
from open_clip import create_model_and_transforms, get_tokenizer
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use a non-GUI backend
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import os

class ImageSearch:
    def __init__(self, embeddings_file, model_name='ViT-B-32', pretrained='openai'):
        """
        Initialize the Image Search Engine.

        Args:
            embeddings_file (str): Path to the file containing precomputed image embeddings.
            model_name (str): Name of the CLIP model to use.
            pretrained (str): Pretrained weights to use for the model.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load precomputed embeddings and filenames
        self.df = pd.read_pickle(embeddings_file)
        self.image_embeddings = np.stack(self.df['embedding'].values)
        self.image_filenames = [os.path.join("coco_images_resized", fname) for fname in self.df['file_name'].values]
        self.filename_to_index = {fname: idx for idx, fname in enumerate(self.image_filenames)}

        # Load CLIP model
        self.model, _, self.preprocess = create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = get_tokenizer(model_name)  # Correctly initialize tokenizer
        self.model = self.model.to(self.device)
        self.model.eval()

    def get_text_embedding(self, text_query):
        """
        Generate a normalized embedding for a text query.

        Args:
            text_query (str): The text query.

        Returns:
            np.ndarray: Normalized text embedding.
        """
        tokenized_text = self.tokenizer([text_query]).to(self.device)  # Use self.tokenizer correctly
        with torch.no_grad():
            embedding = F.normalize(self.model.encode_text(tokenized_text), p=2, dim=1)
        return embedding.cpu().numpy()

    def get_image_embedding(self, image_path):
        """
        Generate a normalized embedding for an image.

        Args:
            image_path (str): Path to the query image.

        Returns:
            np.ndarray: Normalized image embedding.
        """
        image = self.preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = F.normalize(self.model.encode_image(image), p=2, dim=1)
        return embedding.cpu().numpy()

    def find_similar_images(self, query_embedding, top_k=5):
        """
        Find the most similar images for a given query embedding.

        Args:
            query_embedding (np.ndarray): Query embedding.
            top_k (int): Number of top results to return.

        Returns:
            list: List of (filename, similarity score) tuples.
        """
        similarities = np.dot(self.image_embeddings, query_embedding.T).squeeze()
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [(self.image_filenames[i], similarities[i]) for i in top_indices]

    def text_to_image_search(self, text_query, top_k=5):
        """
        Perform a text-to-image search.

        Args:
            text_query (str): The text query.
            top_k (int): Number of top results to return.

        Returns:
            list: List of (filename, similarity score) tuples.
        """
        text_embedding = self.get_text_embedding(text_query)
        return self.find_similar_images(text_embedding, top_k)

    def image_to_image_search(self, image_path, top_k=5):
        """
        Perform an image-to-image search.

        Args:
            image_path (str): Path to the query image.
            top_k (int): Number of top results to return.

        Returns:
            list: List of (filename, similarity score) tuples.
        """
        image_embedding = self.get_image_embedding(image_path)
        return self.find_similar_images(image_embedding, top_k)

    def hybrid_search(self, text_query, image_path, weight=0.5, top_k=5):
        """
        Perform a hybrid search combining text and image queries.

        Args:
            text_query (str): The text query.
            image_path (str): Path to the query image.
            weight (float): Weight for the text query in the hybrid embedding.
            top_k (int): Number of top results to return.

        Returns:
            list: List of (filename, similarity score) tuples.
        """
        text_embedding = self.get_text_embedding(text_query)
        image_embedding = self.get_image_embedding(image_path)

        combined_embedding = F.normalize(
            torch.tensor(weight * text_embedding + (1 - weight) * image_embedding), p=2, dim=1
        ).numpy()

        return self.find_similar_images(combined_embedding, top_k)


RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

def visualize(search_engine, query_embedding, top_results, top_k=5):
    """
    Visualize the embedding space and top-k search results.

    Args:
        search_engine (ImageSearchEngine): The ImageSearchEngine instance.
        query_embedding (np.ndarray): Query embedding used for the search.
        top_results (list): List of (filename, similarity score) tuples for the top-k results.
        top_k (int): Number of top results to display.
    """
    # Reduce dimensions of embeddings for visualization
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(search_engine.image_embeddings)
    reduced_query = pca.transform(query_embedding.reshape(1, -1))

    # Transform the embeddings of the top results
    top_filenames, top_scores = zip(*top_results)
    top_indices = [
        search_engine.filename_to_index[filename] 
        for filename in top_filenames 
        if filename in search_engine.filename_to_index
    ]

    reduced_top = reduced_embeddings[top_indices]


    # Plot embedding space visualization
    plt.figure(figsize=(12, 8))  # Larger figure size for browser
    plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], alpha=0.5, color="gray", label="All Images")
    plt.scatter(reduced_query[0, 0], reduced_query[0, 1], color="red", label="Query", s=150)  # Slightly larger query
    for i, (x, y) in enumerate(reduced_top):
        plt.scatter(x, y, color="blue", label=f"Top-{i+1}" if i == 0 else None)
        plt.text(x, y, f"{top_scores[i]:.2f}", fontsize=10, color="black", ha="center")
    plt.title("Embedding Space Visualization", fontsize=18)
    plt.xlabel("Principal Component 1", fontsize=14)
    plt.ylabel("Principal Component 2", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True)

    embedding_space_path = os.path.join(RESULTS_DIR, "embedding_space.png")
    plt.savefig(embedding_space_path, dpi=150)  # DPI optimized for web
    plt.close()

    # Plot top-k results visualization
    fig, axes = plt.subplots(1, top_k, figsize=(18, 6))  # Ensure images are larger
    for i, ax in enumerate(axes):
        img = plt.imread(top_filenames[i])
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"Top-{i+1}\nSim: {top_scores[i]:.2f}", fontsize=12)

    top_k_results_path = os.path.join(RESULTS_DIR, "top_k_results.png")
    plt.savefig(top_k_results_path, dpi=150)  # DPI optimized for web
    plt.close()

    print(f"Visualizations saved:\n- {embedding_space_path}\n- {top_k_results_path}")
