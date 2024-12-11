import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image

class SpectralNet(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(SpectralNet, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')
        self.pool = tf.keras.layers.MaxPooling2D((2, 2))

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)
        return x

# Fungsi untuk memuat model
@st.cache_resource
def load_custom_model(model_path):
    try:
        custom_objects = {"SpectralNet": SpectralNet}
        # Muat model tanpa kompilasi
        model = load_model(model_path, custom_objects=custom_objects, compile=False)
        # Jika diperlukan, kompile ulang model di sini dengan loss/optimizer yang baru
        model.compile(optimizer='adam', loss='mse')
        return model
    except Exception as e:
        st.error(f"Error saat memuat model: {e}")
        return None

# Fungsi untuk memproses gambar
def process_image(image, model, target_size=(128, 128)):
    try:
        # Resize gambar ke input model
        image = image.resize(target_size)
        # Normalisasi ke [0, 1]
        image_array = np.array(image) / 255.0
        # Tambahkan dimensi batch
        image_array = np.expand_dims(image_array, axis=0)
        
        # Prediksi dengan model
        output = model.predict(image_array)
        return output[0]
    except Exception as e:
        raise RuntimeError(f"Error saat memproses gambar: {e}")

# Path model
MODEL_PATH = "best_model2.h5"  # Sesuaikan dengan lokasi model Anda

# Muat model
model = load_custom_model(MODEL_PATH)

# Judul aplikasi
st.title("Konversi Gambar RGB ke Multispektral")

# Unggah gambar
uploaded_file = st.file_uploader("Unggah gambar RGB", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    try:
        # Baca gambar input
        input_image = Image.open(uploaded_file)
        st.image(input_image, caption="Gambar Input", use_container_width=True)

        # Proses gambar dengan model
        st.write("Memproses gambar...")
        output_image = process_image(input_image, model)

        # Tampilkan output untuk setiap channel multispektral
        st.write("Hasil Prediksi Multispektral:")
        num_bands = output_image.shape[-1]
        for i in range(num_bands):
            st.image(output_image[..., i], caption=f"Band {i + 1}", clamp=True, use_container_width=True)

        # Simpan hasil sebagai file .png
        st.write("Mengunduh hasil...")
        output_image_path = "output_multispectral.png"
        output_image_uint8 = (output_image * 255).astype(np.uint8)
        Image.fromarray(output_image_uint8[..., 0]).save(output_image_path)

        with open(output_image_path, "rb") as file:
            st.download_button(
                label="Unduh Hasil Multispektral",
                data=file,
                file_name="output_multispectral.png",
                mime="image/png",
            )
    except Exception as e:
        st.error(f"Terjadi error: {e}")
else:
    st.info("Unggah gambar RGB untuk memulai.")
