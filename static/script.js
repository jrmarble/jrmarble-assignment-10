document.getElementById("search-form").addEventListener("submit", async function (event) {
    event.preventDefault(); // Prevent form submission

    const textQuery = document.getElementById("text-query").value;
    const imageFile = document.getElementById("image-query").files[0];
    const weight = parseFloat(document.getElementById("weight").value);

    // Validation checks
    if (!textQuery && !imageFile) {
        alert("Please provide at least a text query, an image query, or both.");
        return;
    }

    if (weight < 0.0 || weight > 1.0 || isNaN(weight)) {
        alert("Please enter a weight between 0.0 and 1.0.");
        return;
    }

    // FormData for file upload
    const formData = new FormData();
    if (textQuery) formData.append("text_query", textQuery);
    if (imageFile) formData.append("image_query", imageFile);
    formData.append("weight", weight);

    // Submit the form via Fetch API
    fetch("/search", {
        method: "POST",
        body: formData,
    })
        .then((response) => response.json())
        .then((data) => {
            // Clear and update the results container
            const resultsDiv = document.getElementById("results");
            resultsDiv.innerHTML = ""; // Clear previous results

            if (data.embedding_image || data.top_k_image) {
                if (data.embedding_image) {
                    const embeddingContainer = document.createElement("div");
                    embeddingContainer.className = "result-container";
                    const embeddingImage = document.createElement("img");
                    embeddingImage.src = `/${data.embedding_image}?t=${new Date().getTime()}`; // Prevent caching
                    embeddingImage.alt = "Embedding Space Visualization";
                    embeddingImage.className = "result-image";
                    const embeddingTitle = document.createElement("h2");
                    embeddingTitle.textContent = "Embedding Space Visualization";
                    embeddingContainer.appendChild(embeddingTitle);
                    embeddingContainer.appendChild(embeddingImage);
                    resultsDiv.appendChild(embeddingContainer);
                }

                if (data.top_k_image) {
                    const topKContainer = document.createElement("div");
                    topKContainer.className = "result-container";
                    const topKImage = document.createElement("img");
                    topKImage.src = `/${data.top_k_image}?t=${new Date().getTime()}`; // Prevent caching
                    topKImage.alt = "Top-K Results Visualization";
                    topKImage.className = "result-image";
                    const topKTitle = document.createElement("h2");
                    topKTitle.textContent = "Top-K Results Visualization";
                    topKContainer.appendChild(topKTitle);
                    topKContainer.appendChild(topKImage);
                    resultsDiv.appendChild(topKContainer);
                }
            } else {
                resultsDiv.innerHTML = "<p>No results found.</p>";
            }

            resultsDiv.style.display = "block";
        })
        .catch((error) => {
            console.error("Error performing search:", error);
            alert("An error occurred while performing the search.");
        });
});
