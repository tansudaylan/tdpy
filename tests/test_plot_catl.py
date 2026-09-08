import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import tdpy


class DummyGdat:
    pass


def test_plot_catl_marks_catalog_and_mock_sources():
    gdat = DummyGdat()
    gdat.numbpositext = 2
    gdat.indxsideyposdataflat = np.array([10.0, 20.0])
    gdat.indxsidexposdataflat = np.array([30.0, 40.0])
    gdat.indxdatascorsort = np.array([1, 0])
    gdat.numbsideedge = 3
    gdat.datatype = 'mock'
    gdat.indxsour = np.array([0])
    gdat.indxsoursupn = np.array([1])
    gdat.trueypos = np.array([[5.0, 7.0]])
    gdat.truexpos = np.array([[6.0, 8.0]])
    gdat.truemagtmean = np.array([11.0, 12.0])
    gdat.truemagtstdv = np.array([0.1, 0.2])

    figr, axis = plt.subplots()
    tdpy.plot_catl(gdat, axis, indxsideyposoffs=1, indxsidexposoffs=2)

    text_strings = [text.get_text() for text in axis.texts]

    assert '0' in text_strings
    assert '1' in text_strings
    assert '*' in text_strings
    assert '12, 0.2' in text_strings

    plt.close(figr)