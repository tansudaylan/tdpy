import json
from urllib.parse import parse_qs

import numpy as np
import pytest

from tdpy import mast


def test_mast_request_is_structured_and_connection_is_closed(monkeypatch):
    calls = []

    class Connection:
        def __init__(self, server):
            assert server == 'mast.stsci.edu'

        def request(self, method, path, body, headers):
            calls.append(json.loads(parse_qs(body)['request'][0]))

        def getresponse(self):
            return self

        def getheaders(self):
            return [('Content-type', 'application/json')]

        def read(self):
            return b'{"data": []}'

        def close(self):
            calls.append('closed')

    monkeypatch.setattr(mast.http.client, 'HTTPSConnection', Connection)
    request = {'service': 'Mast.Catalogs.Filtered.Tic', 'params': {'columns': 'ID,rad'}}

    headers, response = mast.quer_mast(request)

    assert calls == [request, 'closed']
    assert headers == [('Content-type', 'application/json')]
    assert json.loads(response) == {'data': []}


def test_tic_crossmatch_requests_mapped_columns_and_retains_solar_units(monkeypatch):
    calls = []

    def query(request):
        calls.append(request)
        return [], json.dumps({'data': [{'ID': 12, 'rad': 0.8, 'mass': 0.9}]})

    monkeypatch.setattr(mast, 'quer_mast', query)
    result = mast.xmat_tici([12], {'ID': 'TICID', 'rad': 'radistar', 'mass': 'massstar'})

    assert calls[0]['params']['columns'] == 'ID,rad,mass'
    assert calls[0]['params']['filters'][0]['values'] == ['12']
    np.testing.assert_array_equal(result['radistar'][0], [0.8])
    assert result['radistar'][1] == r'R$_{\odot}$'
    assert result['massstar'][1] == r'M$_{\odot}$'


def test_explicit_coordinates_do_not_query_the_archive():
    assert mast.setp_coorstrgmast(rasctarg=10.0, decltarg=-20.0) == ('10 -20', 10.0, -20.0)
    with pytest.raises(ValueError):
        mast.setp_coorstrgmast(rasctarg=10.0, strgmast='target')