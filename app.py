
from pathlib import Path

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights


# -----------------------------
# 1. App configuration
# -----------------------------
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "cat_dog_resnet18_best.pth"
DEVICE = torch.device("cpu")


# -----------------------------
# 2. Load the trained model
# -----------------------------
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    class_names = checkpoint.get(
        "class_names", ["cats", "dogs"]
    )

    # Must match the preprocessing used for validation.
    preprocess = ResNet18_Weights.DEFAULT.transforms()

    return model, class_names, preprocess


# -----------------------------
# 3. Page design
# -----------------------------
st.title("🐱 Cat vs Dog Classifier 🐶")

st.write(
    "Upload a cat or dog image and let the trained "
    "ResNet18 deep learning model make a prediction."
)

st.info(
    "This app uses transfer learning with a pretrained ResNet18 model."
)


# -----------------------------
# 4. Upload an image
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)


# -----------------------------
# 5. Predict
# -----------------------------
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        if st.button("Predict Image", type="primary"):
            with st.spinner("Analyzing image..."):
                model, class_names, preprocess = load_model()

                image_tensor = preprocess(image).unsqueeze(0)
                image_tensor = image_tensor.to(DEVICE)

                with torch.inference_mode():
                    outputs = model(image_tensor)
                    probabilities = torch.softmax(outputs, dim=1)[0]

                predicted_index = int(
                    torch.argmax(probabilities).item()
                )

                predicted_class = class_names[predicted_index]
                confidence = probabilities[predicted_index].item() * 100

            # Display a user-friendly class name.
            display_name = {
                "cats": "Cat",
                "dogs": "Dog"
            }.get(predicted_class.lower(), predicted_class)

            st.success(f"Prediction: {display_name} 🐾")
            st.metric(
                label="Model confidence",
                value=f"{confidence:.2f}%"
            )

            st.subheader("Class probabilities")

            for index, class_name in enumerate(class_names):
                name = {
                    "cats": "Cat",
                    "dogs": "Dog"
                }.get(class_name.lower(), class_name)

                probability = probabilities[index].item()
                st.write(f"{name}: {probability * 100:.2f}%")
                st.progress(probability)

            st.caption(
                "Confidence is the model's estimated probability, "
                "not a guarantee that the prediction is correct."
            )

    except Exception as error:
        st.error(f"Something went wrong: {error}")
