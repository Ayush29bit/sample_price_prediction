# sample_price_prediction
Multi-Modal Product analysis and feature extraction using Image and Text Embeddings to predict price of products

# 🧠 Multi-Modal Product Matching using Image & Text Embeddings

A machine learning project that combines **image embeddings** and **text embeddings** to learn product similarity across an e-commerce catalog.

This project extracts image features from product images and text features from product descriptions (`catalog_content`), then combines both representations to train a multi-modal model for product similarity, recommendation, or deduplication.

---

## 📊 Project Overview

Modern e-commerce platforms often list similar or duplicate products with different titles and images.  
This project explores **multi-modal representation learning** to bridge that gap — using both **visual** and **textual** signals.

**Goal:**  
Build a model that understands both what a product *looks like* (image) and *what it’s about* (text), and use those embeddings to compute similarity between items.

---

## 🧩 Dataset

| Column | Description |
|--------|--------------|
| `sample_id` | Unique product ID |
| `catalog_content` | Product description or metadata text |
| `image_link` | URL to the product image |
| `price` | Product price (optional feature) |

---

## 🏗️ Architecture & Approach

### 1️⃣ Image Embeddings
- Model: **Pretrained CNN** (e.g. ResNet50, ViT, or CLIP)
- Framework: `torchvision` / `transformers`
- Output: 2048-dimensional embeddings saved as `.npy` files (`features_part_1.npy`, `features_part_2.npy`, …)

### 2️⃣ Text Embeddings
- Model: **TF-IDF Vectorizer**
- Input: `catalog_content` column
- Output: Sparse vector representing term importance per product description

### 3️⃣ Embedding Fusion
- Combine both embeddings using concatenation:

