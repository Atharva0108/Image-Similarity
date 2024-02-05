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
            reference_features[filename] = (reference_image_gray, hog_features, color_hist)
    return reference_features

# Step 3: Compare New Images with Reference Features
def compare_with_reference(user_image, reference_features):
    # Check if the user image matches with any reference grayscale image
    grayscale_match_found = False
    for _, (ref_image_gray, _, _) in reference_features.items():
        if np.array_equal(user_image, ref_image_gray):
            grayscale_match_found = True
            break

    if not grayscale_match_found:
        st.write("Image does not match with any grayscale reference image. Not-OK")
        return False

    # Proceed with feature matching if grayscale match is found
    resized_user_image = transform.resize(user_image, (128, 128))
    user_features = feature.hog(resized_user_image, pixels_per_cell=(16, 16))
    user_color_hist = np.histogram(user_image, bins=8, range=(0, 1))[0] / 128**2

    hog_match_found = color_match_found = False

    for ref_filename, (ref_image_gray, ref_hog_features, ref_color_hist) in reference_features.items():
        hog_similarity = cosine_similarity([ref_hog_features], [user_features])[0][0]
        if hog_similarity >= hog_threshold:
            hog_match_found = True

        color_similarity = np.sum(np.minimum(ref_color_hist, user_color_hist))
        if color_similarity >= color_threshold:
            color_match_found = True

        if hog_match_found and color_match_found:
            st.write(f"Image matches with {ref_filename}. OK")
            return True

    st.write("Image does not match with any reference image. Not-OK")
    return False


if __name__ == "__main__":
    st.title("Image Similarity Checker")
    st.markdown("---")

    # Dynamic reference folder path for Streamlit Sharing
    reference_folder = os.path.join(os.path.dirname(__file__), "ONS1", "ONS", "reference")
    reference_features = load_reference_features(reference_folder)

    user_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if user_image is not None:
        user_image_array = io.imread(user_image, as_gray=True)  # Load user image as grayscale

        if st.button("Check Similarity"):
            match_found = compare_with_reference(user_image_array, reference_features)
            if match_found:
                st.success("OK")
            else:
                st.error("Not-OK")

