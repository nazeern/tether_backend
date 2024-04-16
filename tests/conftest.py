import pytest
from wsgi import app

@pytest.fixture()
def client():
    return app.test_client()