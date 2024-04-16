

def test_get_status(client):

    res = client.get('/')
    assert b"Success!" in res.data