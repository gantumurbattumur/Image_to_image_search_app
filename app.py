import pandas as pd
import numpy as np
from PIL import Image
import gradio as gr
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import zipfile
from io import BytesIO
import requests
from huggingface_hub import hf_hub_download
# -------------------------
# Load CLIP model
# -------------------------
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")
clip_model.eval()

# -------------------------
# Helper functions
# -------------------------
def get_image_embedding(image: Image.Image):
    """Convert PIL image to normalized CLIP embedding."""
    inputs = clip_processor(images=image, return_tensors="pt")
    with torch.no_grad():
        emb = clip_model.get_image_features(**inputs)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

def search_by_image(query_emb, embeddings, top_k=5):
    """Return indices of top-k most similar embeddings."""
    sims = cosine_similarity([query_emb], embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    top_scores = sims[top_indices]
    return top_indices, top_scores

# -------------------------
# Load dataset + embeddings
# -------------------------
df = pd.read_csv("data/products.csv")
embedding_path = hf_hub_download(
    repo_id="Gantumur/image_to_image_embeddings",
    filename="image_embeddings.npy",
    repo_type="dataset"
)
image_embeddings = np.load(embedding_path)

# -------------------------
# Load the ZIP dataset from Hugging Face
# -------------------------
ZIP_URL = "https://huggingface.co/datasets/Gantumur/ecommerce-images/resolve/main/ecommerce-product-images-18k.zip"
print("⏳ Downloading dataset ZIP from Hugging Face (first run only)...")
zip_bytes = BytesIO(requests.get(ZIP_URL).content)
zip_file = zipfile.ZipFile(zip_bytes)
print("✅ ZIP loaded into memory")

def load_image_from_zip(path_inside_zip):
    """Return PIL image from inside the Hugging Face ZIP file."""
    with zip_file.open(path_inside_zip) as f:
        return Image.open(f).convert("RGB")

# -------------------------
# Gradio search function
# -------------------------
def search_products(uploaded_image, top_k=10):
    query_emb = get_image_embedding(uploaded_image)
    top_indices, _ = search_by_image(query_emb, image_embeddings, top_k)

    # Extract the inner paths after the “.zip!” part
    image_paths = [
    p.split("!")[1].lstrip("/").replace("data/", "")
    for p in df.iloc[top_indices]["image_path"].tolist()
]


    # Load images directly from the ZIP
    images = []
    for p in image_paths:
        try:
            img = load_image_from_zip(p)
            images.append(img)
        except Exception as e:
            print(f"⚠️ Could not load {p}: {e}")
    return images

# -------------------------
# Launch Gradio app
# -------------------------
demo = gr.Interface(
    fn=search_products,
    inputs=gr.Image(type="pil", label="Upload an image"),
    outputs=gr.Gallery(label="Top Matches"),
    title="🛍️ AI Marketplace Search",
    description="Upload an image to find the most visually similar products using CLIP."
)

if __name__ == "__main__":
    demo.launch()
