from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from image_search import visualize, ImageSearch
import numpy as np

app = Flask(__name__)

# Ensure directories exist
UPLOAD_FOLDER = "uploads"
RESULT_FOLDER = "results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Initialize ImageSearchEngine with precomputed embeddings
EMBEDDINGS_FILE = "image_embeddings.pickle"  # Path to your embeddings file
search_engine = ImageSearch(embeddings_file=EMBEDDINGS_FILE)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    # Get query parameters from the form
    text_query = request.form.get('text_query', '').strip()
    image_file = request.files.get('image_query')
    weight = float(request.form.get('weight', 0.5))

    # Validate inputs
    if not text_query and not image_file:
        return jsonify({"error": "Please provide at least a text query, an image query, or both."}), 400
    if not (0.0 <= weight <= 1.0):
        return jsonify({"error": "Weight must be between 0.0 and 1.0."}), 400

    query_embedding = None

    # Process text query (if provided)
    if text_query:
        text_embedding = search_engine.get_text_embedding(text_query)
        query_embedding = weight * text_embedding if query_embedding is None else query_embedding + weight * text_embedding

    # Process image query (if provided)
    if image_file:
        image_path = os.path.join(UPLOAD_FOLDER, image_file.filename)
        image_file.save(image_path)
        image_embedding = search_engine.get_image_embedding(image_path)
        query_embedding = (1 - weight) * image_embedding if query_embedding is None else query_embedding + (1 - weight) * image_embedding

    # Normalize query embedding
    query_embedding = query_embedding / np.linalg.norm(query_embedding)

    # Perform search and visualize results
    top_results = search_engine.find_similar_images(query_embedding, top_k=5)
    visualize(search_engine, query_embedding, top_results, top_k=5)

    # Return paths to generated result images
    embedding_image = "results/embedding_space.png"
    top_k_image = "results/top_k_results.png"

    return jsonify({
        "embedding_image": embedding_image if os.path.exists(embedding_image) else None,
        "top_k_image": top_k_image if os.path.exists(top_k_image) else None,
    })

@app.route('/results/<filename>')
def results(filename):
    return send_from_directory(RESULT_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)

