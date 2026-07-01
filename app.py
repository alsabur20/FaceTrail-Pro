import streamlit as st
import os
import zipfile
import numpy as np
import cv2
from sklearn.cluster import DBSCAN
import shutil
from insightface.app import FaceAnalysis

# --- UI CONFIG ---
st.set_page_config(page_title="FaceTrail Pro", page_icon="📸", layout="wide")
st.title("📸 FaceTrail Pro")
st.write("Upload photos and automatically group people across your event.")

# --- SESSION STATE ---
if "clusters" not in st.session_state:
    st.session_state.clusters = None

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# --- MODEL ---
@st.cache_resource
def load_model():
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)
    return app

model = load_model()

# --- UPLOADER ---
uploaded_files = st.file_uploader(
    "Upload Event Photos",
    accept_multiple_files=True,
    type=["jpg", "jpeg", "png"],
    key=f"uploader_{st.session_state.uploader_key}"
)

# --- PROCESS BUTTON ---
if uploaded_files and st.session_state.clusters is None:
    if st.button("🚀 Process & Group Photos", type="primary"):

        progress_text = st.empty()
        progress_bar = st.progress(0)

        image_paths = []
        embeddings = []
        face_map = []
        thumbnails = []

        os.makedirs("temp", exist_ok=True)

        # ---- PROCESS IMAGES ----
        for i, file in enumerate(uploaded_files):
            progress_text.text(f"Processing photo {i+1} of {len(uploaded_files)}: {file.name}")

            path = os.path.join("temp", file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            img = cv2.imread(path)

            # Slight upscale (better detection)
            img = cv2.resize(img, None, fx=1.2, fy=1.2)

            faces = model.get(img)

            for face in faces:
                embeddings.append(face.embedding)
                face_map.append((file.name, file.getvalue()))

                x1, y1, x2, y2 = map(int, face.bbox)
                crop = img[y1:y2, x1:x2]
                crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                thumbnails.append(crop)

            progress_bar.progress((i + 1) / len(uploaded_files))

        if not embeddings:
            progress_text.empty()
            progress_bar.empty()
            st.warning("No faces detected.")
            st.stop()

        # ---- CLUSTERING ----
        progress_text.text("Grouping similar faces...")

        embeddings = np.array(embeddings)

        clustering = DBSCAN(eps=0.7, min_samples=1, metric="cosine").fit(embeddings)
        labels = clustering.labels_

        clusters = {}

        for label, (filename, filebytes), face_img in zip(labels, face_map, thumbnails):
            person = f"Person {label + 1}"

            if person not in clusters:
                clusters[person] = {
                    "faces": [],
                    "photos": {}
                }

            clusters[person]["faces"].append(face_img)
            clusters[person]["photos"][filename] = filebytes

        st.session_state.clusters = clusters

        progress_text.empty()
        progress_bar.empty()

        st.rerun()

# --- RESULTS UI ---
if st.session_state.clusters:

    st.success("Faces successfully grouped across your photos!")

    if st.button("🧹 Clear Results"):
        st.session_state.clusters = None
        st.session_state.uploader_key += 1
        st.rerun()

    st.divider()
    st.subheader("Results")

    for person, data in st.session_state.clusters.items():

        with st.container():
            col1, col2, col3 = st.columns([1, 1, 2])

            with col1:
                st.write(f"### {person}")
                st.caption(f"{len(data['photos'])} photos")

            with col2:
                st.image(data["faces"][0], width=120)

            with col3:
                zip_buffer = zipfile.ZipFile(
                    f"{person}.zip", "w", zipfile.ZIP_DEFLATED
                )

                mem_zip = zipfile.ZipFile(
                    os.path.join("temp.zip"), "w", zipfile.ZIP_DEFLATED
                )

                buffer = zipfile.ZipFile(
                    os.path.join("temp_mem.zip"), "w", zipfile.ZIP_DEFLATED
                )

                zip_bytes = zipfile.ZipFile(
                    os.path.join("temp_mem2.zip"), "w"
                )

                # clean memory zip
                import io
                zip_io = io.BytesIO()
                with zipfile.ZipFile(zip_io, "w", zipfile.ZIP_DEFLATED) as zipf:
                    for fname, fbytes in data["photos"].items():
                        zipf.writestr(fname, fbytes)

                st.download_button(
                    label=f"Download {person}",
                    data=zip_io.getvalue(),
                    file_name=f"{person.replace(' ', '_')}.zip",
                    mime="application/zip",
                    use_container_width=True
                )

        st.divider()

st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px; padding: 10px 0;'>
    FaceTrail Pro • AI Face Clustering • Built with Streamlit  
</div>
""", unsafe_allow_html=True)
