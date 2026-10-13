"""MAST requests, TIC cross-matches, and coordinate resolution."""

import http.client
import json
import sys
from urllib.parse import urlencode

import numpy as np
from .verbosity import print


def quer_mast(request):
    """Send a structured MAST request and return headers and decoded response text."""

    connection = http.client.HTTPSConnection('mast.stsci.edu')
    headers = {'Content-type': 'application/x-www-form-urlencoded', 'Accept': 'text/plain',
               'User-agent': 'python-requests/' + '.'.join(map(str, sys.version_info[:3]))}
    print('Reading from https://mast.stsci.edu/api/v0/invoke...')
    try:
        connection.request('POST', '/api/v0/invoke', urlencode({'request': json.dumps(request)}), headers)
        response = connection.getresponse()
        return response.getheaders(), response.read().decode('utf-8')
    finally:
        connection.close()


def xmat_tici(listtici, dictnamefeatmast):
    """Retrieve the requested TIC columns as arrays paired with unit labels."""

    identifiers = np.asarray(listtici).astype(str).tolist()
    if not identifiers:
        raise ValueError('Provide at least one TIC identifier')
    request = {'service': 'Mast.Catalogs.Filtered.Tic', 'format': 'json',
               'params': {'columns': ','.join(dictnamefeatmast),
                          'filters': [{'paramName': 'ID', 'values': identifiers}]}}
    _, content = quer_mast(request)
    rows = json.loads(content)['data']
    labels = {'radistar': r'R$_{\odot}$', 'massstar': r'M$_{\odot}$',
              'rascstar': 'degree', 'declstar': 'degree', 'tmptstar': 'K',
              'magtsystTESS': 'mag'}
    return {name: [np.asarray([row[column] for row in rows]), labels.get(name, '')]
            for column, name in dictnamefeatmast.items()}


def retr_listtablobsv(strgmast):
    """Retrieve TESS time-series observation records for a named target."""

    from astroquery.mast import Observations

    if strgmast is None:
        raise ValueError('strgmast must identify a target')
    print('Reading from https://mast.stsci.edu/...')
    return Observations.query_criteria(obs_collection='TESS', dataproduct_type='timeseries',
                                       objectname=strgmast)


def retr_rascdeclfromstrgmast(strgmast):
    """Return coordinates in degrees for the first TIC match to a target name."""

    from astroquery.mast import Catalogs

    print('Reading from https://mast.stsci.edu/...')
    catalog = Catalogs.query_object(strgmast, catalog='TIC')
    if len(catalog) == 0:
        raise ValueError('No TIC match was returned')
    return catalog[0]['ra'], catalog[0]['dec']


def setp_coorstrgmast(rasctarg=None, decltarg=None, strgmast=None):
    """Resolve a target name or format supplied coordinates in degrees."""

    if ((strgmast is not None and (rasctarg is not None or decltarg is not None))
            or (strgmast is None and (rasctarg is None or decltarg is None))):
        raise ValueError('Provide a target name or both coordinates, not both')
    if rasctarg is None:
        rasctarg, decltarg = retr_rascdeclfromstrgmast(strgmast)
    if strgmast is None:
        strgmast = '%g %g' % (rasctarg, decltarg)
    return strgmast, rasctarg, decltarg