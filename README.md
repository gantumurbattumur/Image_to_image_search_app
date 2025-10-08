# 🛍️ Image to image search app

This is an **image-to-image search app** powered by [CLIP (Contrastive Language–Image Pretraining)](https://openai.com/research/clip).  
Upload any product image, and the app finds visually similar items from an e-commerce dataset.

---

## 🚀 Features
- Uses **OpenAI’s CLIP model** (`clip-vit-base-patch16`) to embed product images.
- Performs **cosine similarity** search between embeddings.
- Loads images **directly from a Hugging Face ZIP dataset** (no need to unzip locally).
- Built with **Gradio** for easy web deployment.

---

## 🧠 How It Works
1. The uploaded image is converted into a CLIP embedding.
2. Pre-computed embeddings from the dataset (`models/image_embeddings.npy`) are compared using cosine similarity.
3. The top visually similar products are retrieved and displayed.

---

## 🛠️ Setup (Local)

```bash
git clone https://github.com/<your-username>/ai_marketplace.git
cd ai_marketplace
pip install -r requirements.txt
python app.py
```
Then open http://127.0.0.1:7860 in your browser.