import io

from pytest_mock import MockerFixture

from validators import ImageValidator

# from validators import ImageValidator


def test_predict_image_no_image_file(client):

    res = client.post('/predict')

    assert res.status_code == 400
    data = res.get_json()

    assert "Image file not provided in form data" in data.get('error')


def test_predict_image_wrong_file_type(client):

    image_filepath = "tests/data/test_bun.txt"
    data = {
        'image': (io.BytesIO(b"some initial text data"), image_filepath)
    }
    res = client.post('/predict', data=data)

    assert res.status_code == 400
    data = res.get_json()

    assert "Image must be in JPEG format" in data.get('error')


def test_predict_image_improper_image_dimensions(
    client,
    mocker: MockerFixture,
):
    
    def mock_image_validator(*args, **kwargs):
        return ImageValidator(xmax=1, ymax=1)
    
    mocker.patch("wsgi.ImageValidator", new=mock_image_validator)

    image_filepath = "tests/data/test_bun.jpeg"
    data = {
        'image': (open(image_filepath, 'rb'))
    }
    res = client.post('/predict', data=data)

    assert res.status_code == 400
    data = res.get_json()

    assert "Image exceeds maximum dimensions" in data.get('error')


def test_predict_bun_image(client):

    image_filepath = "tests/data/test_bun.jpeg"
    data = {
        'image': (open(image_filepath, 'rb'))
    }
    res = client.post('/predict', data=data)

    assert res.status_code == 200
    data = res.get_json()

    assert data['response'] == 0


def test_predict_hungry_cat_image(client):

    image_filepath = "tests/data/test_hungry_cat.jpeg"
    data = {
        'image': (open(image_filepath, 'rb'))
    }
    res = client.post('/predict', data=data)

    assert res.status_code == 200
    data = res.get_json()

    assert data['response'] == 1