from deepface import DeepFace

def cache_all_models():
    print("Starting Deepface model pre-download")

    # Face Recognition Models
    models = ["VGG-Face"]

    for model in models:
        try:
            print(f"Downloading model weights for {model}")
            DeepFace.build_model(model)
        except Exception as e:
            print(f"Error caching {model}: {str(e)}")

    # Facial Attribute/Analysis Models
    attributes = ["Age", "Gender", "Emotion", "Race"]
    for attribute in attributes:
        try:
            print(f"Downloading attributes weights for {attribute}")
            DeepFace.build_model(attribute, 'facial_attribute')
        except Exception as e:
            print(f"Error caching {attribute}: {str(e)}")

    print("All models downloaded successfully")

if __name__ == "__main__":
    cache_all_models()