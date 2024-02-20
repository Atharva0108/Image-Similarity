import os
import numpy as np
import streamlit as st
from skimage import io, transform, feature, color
from sklearn.metrics.pairwise import cosine_similarity

# Step 1: Define Thresholds
hog_threshold = 0.95  # Threshold for HOG feature comparison
color_threshold = 0.9  # Threshold for color histogram comparison

# Step 2: Load Reference Images and Extract Features
@st.cache
def load_reference_features(reference_folder):
    reference_features = {}
    for filename in os.listdir(reference_folder):
        if filename.lower().endswith((".jpg", ".png")):
            filepath = os.path.join(reference_folder, filename)
            # Load reference image in color
            reference_image = io.imread(filepath)
            # Convert reference image to grayscale
            reference_image_gray = color.rgb2gray(reference_image)

            # Resize reference image
            resized_image = transform.resize(reference_image_gray, (128, 128))
            hog_features = feature.hog(resized_image, pixels_per_cell=(16, 16))
            color_hist = np.histogram(reference_image_gray, bins=8, range=(0, 1))[0] / 128**2
            reference_features[filename] = (reference_image, reference_image_gray, hog_features, color_hist)
    return reference_features

def compare_with_reference(user_image, user_image_gray, user_features, user_color_hist, reference_features):
    match_found = False
    for ref_filename, (ref_image, ref_image_gray, ref_hog_features, ref_color_hist) in reference_features.items():
        # Check grayscale match
        if np.array_equal(user_image_gray, ref_image_gray):
            match_found = True
            st.write(f"Grayscale match found with {ref_filename}.")
            # Compare HOG features
            hog_similarity = cosine_similarity([ref_hog_features], [user_features])[0][0]
            color_similarity = np.sum(np.minimum(ref_color_hist, user_color_hist))

            if hog_similarity >= hog_threshold and color_similarity >= color_threshold:
                st.write(f"Image matches with {ref_filename}. OK")
                match_found = True
                
                # Display images side by side
                col1, col2 = st.columns(2)
                with col1:
                    st.image(ref_image, caption=f"Reference Image: {ref_filename}")
                with col2:
                    st.image(user_image, caption='User Image')
                
            else:
                st.write(f"Image does not match with {ref_filename}. Not-OK")
                # Highlight the areas of mismatch
                highlight_mismatch(user_image, ref_image, user_image_gray, ref_image_gray)
            break

    if not match_found:
        st.write("Image does not match with any reference image. Not-OK")


def highlight_mismatch(user_image, ref_image, user_image_gray, ref_image_gray):
    # You can implement the highlighting logic here
    pass

if __name__ == "__main__":
    st.title("Image Similarity Checker")
    st.markdown("---")

    # Dynamic reference folder path for Streamlit Sharing
    reference_folder = os.path.join(os.path.dirname(__file__), "ONS1", "ONS", "reference")
    reference_features = load_reference_features(reference_folder)

    user_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if user_image is not None:
        user_image_array = io.imread(user_image)  # Load user image
        user_image_gray = color.rgb2gray(user_image_array)
        resized_user_image = transform.resize(user_image_gray, (128, 128))
        user_features = feature.hog(resized_user_image, pixels_per_cell=(16, 16))
        user_color_hist = np.histogram(user_image_gray, bins=8, range=(0, 1))[0] / 128**2

        if st.button("Check Similarity"):
            compare_with_reference(user_image_array, user_image_gray, user_features, user_color_hist, reference_features)
