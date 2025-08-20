import requests

# Update this URL to your API endpoint if different
url = "http://localhost:8000/predict"


def test_valid_image():
    image_path = (
        "/home/starlord/Garbage-classification-mlops/"
        "subset_data/test/cardboard/cardboard_00018.jpg"
    )
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = requests.post(
        url,
        data=image_bytes,
        headers={"Content-Type": "application/octet-stream"},
    )
    print("Valid image response:", response.status_code, response.json())


def test_invalid_image():
    # Prepare invalid (non-image) data
    data = b"this is not an image"
    response = requests.post(
        url, data=data, headers={"Content-Type": "application/octet-stream"}
    )
    print("Invalid image response:", response.status_code, response.json())


if __name__ == "__main__":
    test_valid_image()
    test_invalid_image()
