from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    json_data = response.json()
    assert "message" in json_data


def test_prediction_endpoint():
    image_path = (
        "/home/starlord/Garbage-classification-mlops/"
        "subset_data/test/cardboard/cardboard_00018.jpg"
    )
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    response = client.post(
        "/predict",
        content=image_bytes,
        headers={"Content-Type": "application/octet-stream"}
    )
    assert response.status_code == 200
    json_data = response.json()
    assert "predicted_class" in json_data
    assert "confidence" in json_data
    assert isinstance(json_data["predicted_class"], int)
    assert 0.0 <= json_data["confidence"] <= 1.0


def test_invalid_input():
    bad_data = b"this is not an image"
    response = client.post(
        "/predict",
        content=bad_data,
        headers={"Content-Type": "application/octet-stream"},
    )

    assert response.status_code == 400
