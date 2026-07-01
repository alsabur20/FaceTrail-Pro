# 📸 FaceTrail Pro

FaceTrail Pro is an AI-powered face clustering application that automatically groups people across a collection of photos and allows you to download all images for each individual.

---

## 🚀 Features

* 🔍 Detects faces in uploaded images using state-of-the-art models
* 🧠 Groups faces by identity using clustering algorithms
* 👤 Automatically organizes photos by person
* 📦 Download images of each person as a ZIP file
* ⚡ Clean and responsive Streamlit UI
* 🔄 Real-time progress tracking during processing

---

## 🛠️ Tech Stack

* **Frontend/UI:** Streamlit
* **Face Detection & Embeddings:** InsightFace (RetinaFace + ArcFace)
* **Clustering:** DBSCAN (cosine similarity)
* **Image Processing:** OpenCV, PIL

---

## 📂 How It Works

1. Upload a set of images
2. The app detects all faces in each image
3. Extracts embeddings for each face
4. Clusters similar faces together
5. Groups photos by detected individuals
6. Allows downloading each person's photos as a ZIP

---

## 🧪 Running Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 🌐 Deployment

This app can be deployed on platforms like:

* Hugging Face Spaces (recommended)
* Railway

---

## ⚠️ Notes

* Performance depends on number and size of images
* Large batches may take longer to process
* Best results with clear, front-facing images

---

## 📌 Future Improvements

* Face naming and labeling
* Photo gallery preview per person
* Improved clustering accuracy
* Cloud storage integration

---

## 👨‍💻 Author

Abdul Sabur
Final Year BS Computer Science Student

---

## ⭐️ Show Support

If you like this project, consider giving it a star ⭐ on GitHub!
