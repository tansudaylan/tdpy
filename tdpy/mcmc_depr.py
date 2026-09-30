# numerics
import numpy as np

import scipy as sp
from scipy.special import erfi
import scipy.fftpack
import scipy.stats

# plotting
import matplotlib as mpl
from matplotlib import pyplot as plt
import multiprocessing

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

# utilities
import os

# astropy
import astropy.coordinates, astropy.units
import astropy.io

import sklearn

import tdpy.util
from .util import *

def plot_propeffi(path, numbswep, numbpara, listaccp, listindxparamodi, strgpara):

    indxlistaccp = np.where(listaccp == True)[0]
    binstime = np.linspace(0., numbswep - 1., 10)
    
    numbcols = 2
    numbrows = (numbpara + 1) / 2
    figr, axgr = plt.subplots(numbrows, numbcols, figsize=(16, 4 * (numbpara + 1)))
    if numbrows == 1:
        axgr = [axgr]
    for a, axrw in enumerate(axgr):
        for b, axis in enumerate(axrw):
            k = 2 * a + b
            if k == numbpara:
                axis.axis('off')
                break
            indxlistpara = np.where(listindxparamodi == k)[0]
            indxlistintc = intersect1d(indxlistaccp, indxlistpara, assume_unique=True)
            histotl = axis.hist(indxlistpara, binstime, color='b')
            histaccp = axis.hist(indxlistintc, binstime, color='g')
            axis.set_title(strgpara[k])
    figr.subplots_adjust(hspace=0.3)
    figr.savefig(path + 'propeffi.pdf')
    plt.close(figr)


def plot_trac(path, listpara, labl, truepara=None, scalpara='self', titl=None, \
                        boolquan=True, listvarbdraw=None, listlabldraw=None, numbbinsplot=20, logthist=False, listcolrdraw=None):
    
    if not np.isfinite(listpara).all():
        return
    
    if not np.isfinite(listpara).all():
        raise Exception('')
    
    if listpara.size == 0:
        return

    maxmpara = np.amax(listpara)
    if scalpara == 'logt':
        minmpara = np.amin(listpara[np.where(listpara > 0.)])
        bins = icdf_logt(np.linspace(0., 1., numbbinsplot + 1), minmpara, maxmpara)
    else:
        minmpara = np.amin(listpara)
        bins = icdf_self(np.linspace(0., 1., numbbinsplot + 1), minmpara, maxmpara)
    limspara = np.array([minmpara, maxmpara])
        
    if boolquan:
        quanarry = sp.stats.mstats.mquantiles(listpara, prob=[0.025, 0.16, 0.84, 0.975])

    if scalpara == 'logt':
        numbtick = 5
        listtick = np.logspace(np.log10(minmpara), np.log10(maxmpara), numbtick)
        listlabltick = ['%.3g' % tick for tick in listtick]
    
    figr, axrw = plt.subplots(1, 2, figsize=(14, 7))
    if titl is not None:
        figr.suptitle(titl, fontsize=18)
    for n, axis in enumerate(axrw):
        if n == 0:
            axis.plot(listpara, lw=0.5)
            axis.set_xlabel('$i_{samp}$')
            axis.set_ylabel(labl)
            if truepara is not None and not np.isnan(truepara):
                axis.axhline(y=truepara, color='g', lw=4)
            if scalpara == 'logt':
                axis.set_yscale('log')
                axis.set_yticks(listtick)
                axis.set_yticklabels(listlabltick)
            axis.set_ylim(limspara)
            if listvarbdraw is not None:
                for k in range(len(listvarbdraw)):
                    axis.axhline(listvarbdraw[k], label=listlabldraw[k], color=listcolrdraw[k], lw=3)
            if boolquan:
                axis.axhline(quanarry[0], color='b', ls='--', lw=2)
                axis.axhline(quanarry[1], color='b', ls='-.', lw=2)
                axis.axhline(quanarry[2], color='b', ls='-.', lw=2)
                axis.axhline(quanarry[3], color='b', ls='--', lw=2)
        else:
            axis.hist(listpara, bins=bins)
            axis.set_xlabel(labl)
            if logthist:
                axis.set_yscale('log')
            axis.set_ylabel('$N_{samp}$')
            if truepara is not None and not np.isnan(truepara):
                axis.axvline(truepara, color='g', lw=4)
            if scalpara == 'logt':
                axis.set_xscale('log')
                axis.set_xticks(listtick)
                axis.set_xticklabels(listlabltick)
            axis.set_xlim(limspara)
            if listvarbdraw is not None:
                for k in range(len(listvarbdraw)):
                    axis.axvline(listvarbdraw[k], label=listlabldraw[k], color=listcolrdraw[k], lw=3)
            if boolquan:
                axis.axvline(quanarry[0], color='b', ls='--', lw=2)
                axis.axvline(quanarry[1], color='b', ls='-.', lw=2)
                axis.axvline(quanarry[2], color='b', ls='-.', lw=2)
                axis.axvline(quanarry[3], color='b', ls='--', lw=2)
                
    figr.subplots_adjust()#top=0.9, wspace=0.4, bottom=0.2)

    figr.savefig(path + '_trac.pdf')
    plt.close(figr)


def plot_plot(path, xdat, ydat, lablxdat, lablydat, scalxaxi, titl=None, linestyl=[None], colr=[None], legd=[None], **args):
    
    if not isinstance(ydat, list):
        listydat = [ydat]
    else:
        listydat = ydat

    figr, axis = plt.subplots(figsize=(6, 6))
    for k, ydat in enumerate(listydat):
        if k == 0:
            linestyl = '-'
        else:
            linestyl = '--'
        axis.plot(xdat, ydat, ls=linestyl, color='k', **args)
        # temp
        #axis.plot(xdat, ydat, ls=linestyl[k], color=colr[k], label=legd[k], **args)
    axis.set_ylabel(lablydat)
    axis.set_xlabel(lablxdat)
    if scalxaxi == 'logt':
        axis.set_xscale('log')
    if titl is not None:
        axis.set_title(titl)
    plt.tight_layout()
    figr.savefig(path + '.pdf')
    plt.close(figr)


def plot_hist(path, listvarb, strg, titl=None, numbbins=20, truepara=None, boolquan=True, typefileplot='pdf', \
                                            scalpara='self', listvarbdraw=None, listlabldraw=None, listcolrdraw=None):

    minmvarb = np.amin(listvarb)
    maxmvarb = np.amax(listvarb)
    if scalpara == 'logt':
        bins = icdf_logt(np.linspace(0., 1., numbbins + 1), minmvarb, maxmvarb)
    else:
        bins = icdf_self(np.linspace(0., 1., numbbins + 1), minmvarb, maxmvarb)
    figr, axis = plt.subplots(figsize=(6, 6))
    axis.hist(listvarb, bins=bins)
    axis.set_ylabel(r'$N_{samp}$')
    axis.set_xlabel(strg)
    if truepara is not None:
        axis.axvline(truepara, color='g', lw=4)
    if listvarbdraw is not None:
        for k in range(len(listvarbdraw)):
            axis.axvline(listvarbdraw[k], label=listlabldraw[k], color=listcolrdraw[k], lw=3)
    if boolquan:
        quanarry = sp.stats.mstats.mquantiles(listvarb, prob=[0.025, 0.16, 0.84, 0.975])
        axis.axvline(quanarry[0], color='b', ls='--', lw=2)
        axis.axvline(quanarry[1], color='b', ls='-.', lw=2)
        axis.axvline(quanarry[2], color='b', ls='-.', lw=2)
        axis.axvline(quanarry[3], color='b', ls='--', lw=2)
    if titl is not None:
        axis.set_title(titl)
    plt.tight_layout()
    figr.savefig(path + '_hist.%s' % typefileplot)
    plt.close(figr)


def retr_limtpara(scalpara, minmpara, maxmpara, meanpara, stdvpara):
    
    numbpara = len(scalpara)
    limtpara = np.empty((2, numbpara))
    indxpara = np.arange(numbpara)
    for n in indxpara:
        if scalpara[n] == 'self':
            limtpara[0, n] = minmpara[n]
            limtpara[1, n] = maxmpara[n]
        if scalpara[n] == 'gaus':
            limtpara[0, n] = meanpara[n] - 10 * stdvpara[n]
            limtpara[1, n] = meanpara[n] + 10 * stdvpara[n]
    
    return limtpara


def retr_lpos(para, *dictlpos):
     
    gdat, indxpara, scalpara, minmpara, maxmpara, meangauspara, stdvgauspara, retr_llik, retr_lpri = dictlpos
    
    boolreje = False
    for k in indxpara:
        if scalpara[k] != 'gaus':
            if para[k] < minmpara[k] or para[k] > maxmpara[k]:
                lpos = -np.inf
                boolreje = True
    
    if not boolreje:
        llik = retr_llik(para, gdat)
        lpri = 0.
        if retr_lpri is None:
            for k in indxpara:
                if scalpara[k] == 'gaus':
                    lpri += (para[k] - meangauspara[k]) / stdvgauspara[k]**2
        else:
            lpri = retr_lpri(para, gdat)
        lpos = llik + lpri
    
    #print('lpos')
    #print(lpos)
    #print('')
    
    return lpos


def opti(pathimag, retr_llik, minmpara, maxmpara, numbtopp=3, numbiter=5):

    numbsamp = 4
    indxsamp = np.arange(numbsamp)
    numbpara = minmpara.size
    indxiter = np.arange(numbiter)
    indxtopp = np.arange(numbtopp)

    # seeds
    listfact = []
    listparacent = []
    listopen = []
    listllikmaxmseed = []
    # all samples
    #listllik = np.empty(0)
    #listpara = np.empty((0, numbpara))
    
    for i in indxiter:
        print('i')
        print(i)
        #print('listllik')
        #print(listllik)
        #print('listpara')
        #print(listpara)
        
        print('listfact')
        print(listfact)
        print('listparacent')
        print(listparacent)
        print('listllikmaxmseed')
        print(listllikmaxmseed)
        print('listopen')
        print(listopen)
        if i == 0:
            minmparatemp = minmpara
            maxmparatemp = maxmpara
            thisindxseed = 0
            paramidi = (maxmpara + minmpara) / 2.
            listfact.append(1.)
            listparacent.append(paramidi)
            listopen.append(True)
            listllikmaxmseed.append([])
        else:
            indxopen = np.where(listopen)[0]
            thisindxseed = np.random.choice(indxopen)
            print('thisindxseed')
            print(thisindxseed)
            maxmparatemp = listparacent[thisindxseed] + listfact[thisindxseed] * (maxmpara - listparacent[thisindxseed])
            minmparatemp = listparacent[thisindxseed] - listfact[thisindxseed] * (listparacent[thisindxseed] - minmpara)
        print('thisindxseed')
        print(thisindxseed)
        para = np.random.rand(numbpara * numbsamp).reshape((numbsamp, numbpara)) * (maxmpara[None, :] - minmpara[None, :]) + minmpara[None, :]
        print('para')
        print(para)
        
        print('Evaluating samples...')
        llik = np.empty(numbsamp)
        for k in indxsamp:
            llik[k] = retr_llik(para[k, :])
        print('llik')
        print(llik)
        listllikmaxmseed[thisindxseed] = np.amax(llik)
        
        # add new seeds
        if i == 0:
            for k in indxtopp:
                # factor
                listfact.append(0.5 * listfact[thisindxseed])
                
                # parameters of the seeds
                indxsampsort = np.argsort(llik)[::-1]
                listparacent.append(para[indxsampsort, :])
                
                # determine if still open
                boolopen = np.amax(llik) >= listllikmaxmseed[thisindxseed]
                listopen.append(boolopen)
                
                listllikmaxmseed.append([])

        #listllik = np.concatenate((listllik, llik))
        #listpara = np.concatenate((listpara, para), 0)
        
        #print('listllik')
        #print(listllik)
        #print('listpara')
        #print(listpara)
        print('listfact')
        print(listfact)
        print('listparacent')
        print(listparacent)
        print('listllikmaxmseed')
        print(listllikmaxmseed)
        print('listopen')
        print(listopen)
        print('')
        print('')
        print('')
        if not np.array(listopen).any():
            break
    
    return listparatopp


def samp(gdat, pathimag, numbsampwalk, retr_llik, \
              # model parameters
              ## list of names of parameters
              listnamepara, \
              ## list of labels of parameters
              listlablpara, \
              ## list of scalings of parameters
              scalpara, \
              ## list of minima of parameters
              minmpara, \
              ## list of maxima of parameters
              maxmpara, \
              meangauspara=None, \
              stdvgauspara=None, \
              retr_lpri=None, \
              # Boolean flag to turn on multiprocessing
              boolmult=True, \
              # burn-in
              ## number of samples in a precursor run whose final state will be used as the initial state of the actual sampler
              numbsampburnwalkinit=0, \
              ## number of initial samples to be burned
              numbsampburnwalk=0, \
              # derivation
              retr_dictderi=None, \
              listlablparaderi=None, \
              diagmode=True, strgextn='', typesamp='emce', typefileplot='png', verbtype=1, strgsaveextn=None):
    if typesamp == 'nest':
        parameter_count = len(listnamepara)
        initial = np.asarray([(np.asarray(minmpara) + np.asarray(maxmpara)) / 2.])
        walker_count = max(20, 2 * parameter_count)
        chain, logprob, evidence = tdpy.util._pcat_legacy_chains(
            gdat, retr_llik, retr_lpri, listnamepara, scalpara,
            np.asarray(minmpara), np.asarray(maxmpara), meangauspara,
            stdvgauspara, initial, walker_count, numbsampwalk,
            numbsampburnwalkinit, pathimag, verbtype,
            estimate_log_evidence=True,
        )
        retained = chain[:, numbsampburnwalk:, :].reshape(-1, parameter_count)
        parameters = {name: retained[:, index] for index, name in enumerate(listnamepara)}
        parameters['lpos'] = logprob[:, numbsampburnwalk:].ravel()
        derived = {'log_evidence': evidence['log_evidence'],
                   'log_evidence_relative_error': evidence['relative_error'],
                   'evidence_effective_sample_size': evidence['effective_sample_size']}
        return parameters, derived

    numbpara = len(listlablpara)
   
    if numbsampwalk <= numbsampburnwalk:
        raise Exception('Burn-in samples cannot outnumber samples.')
    
    if isinstance(minmpara, list):
        minmpara = np.array(minmpara)

    if isinstance(maxmpara, list):
        maxmpara = np.array(maxmpara)

    if numbpara != minmpara.size:
        raise Exception('')
    if numbpara != maxmpara.size:
        raise Exception('')

    indxpara = np.arange(numbpara)
    
    if typesamp == 'emce':
        numbwalk = max(20, 2 * numbpara)
        indxwalk = np.arange(numbwalk)
        numbsamptotl = numbsampwalk * numbwalk

    # plotting
    ## plot limits 
    limtpara = retr_limtpara(scalpara, minmpara, maxmpara, meangauspara, stdvgauspara)

    ## plot bins
    numbbins = 20
    indxbins = np.arange(numbbins)
    binspara = np.empty((numbbins + 1, numbpara)) 
    for k in indxpara:
        binspara[:, k] = np.linspace(limtpara[0, k], limtpara[1, k], numbbins + 1)
    meanpara = (binspara[1:, :] + binspara[:-1, :]) / 2.
    
    for k in indxpara:
        if minmpara[k] >= maxmpara[k]:
            raise Exception('')
    
    dictlpos = [gdat, indxpara, scalpara, minmpara, maxmpara, meangauspara, stdvgauspara, retr_llik, retr_lpri]
    

    if verbtype == 2:
        print('scalpara')
        print(scalpara)
        print('minmpara')
        print(minmpara)
        print('maxmpara')
        print(maxmpara)
        print('meangauspara')
        print(meangauspara)
        print('stdvgauspara')
        print(stdvgauspara)
        print('limtpara')
        print(limtpara)
    
    # initialize
    if strgsaveextn is None or not os.path.exists(strgsaveextn):
        parainitcent = np.empty(numbpara)
        for m in indxpara:
            if scalpara[m] == 'self':
                parainitcent[m]  = limtpara[0, m] + 0.5 * (limtpara[1, m] - limtpara[0, m])
    else:
        print('Reading the initial state from %s...' % strgsaveextn)
        parainitcent = np.loadtxt(strgsaveextn)
    parainit = [np.empty(numbpara) for k in indxwalk]
    for m in indxpara:
        for k in indxwalk:
            if scalpara[m] == 'self':
                stdvinit = 10.
                parainit[k][m] = 0.5 / stdvinit * scipy.stats.truncnorm.rvs(-stdvinit, stdvinit) * (limtpara[1, m] - limtpara[0, m]) + parainitcent[m]
            if scalpara[m] == 'gaus':
                parainit[k][m] = np.random.normal(meangauspara[m], stdvgauspara[m])
        
    if typesamp == 'emce':
        if verbtype > 0:
            progress = True
        else:
            progress = False
    
        numbsamp = numbwalk * numbsampwalk
        indxsampwalk = np.arange(numbsampwalk)
        indxsamp = np.arange(numbsamp)
        numbsampburn = numbsampburnwalkinit * numbwalk
        if diagmode:
            if numbsampwalk == 0:
                raise Exception('')
    
        listparafittwalk, listlposwalk = tdpy.util._pcat_legacy_chains(
            gdat, retr_llik, retr_lpri, listnamepara, scalpara,
            minmpara, maxmpara, meangauspara, stdvgauspara,
            parainit, numbwalk, numbsampwalk,
            numbsampburnwalkinit, pathimag, verbtype,
        )
        
        # get rid of burn-in and thin
        numbavail = numbsampwalk - numbsampburnwalk
        if numbavail <= 0:
            raise ValueError('No post-burn-in samples remain to retain.')
        numbkeep = min(numbsampwalk, numbavail)
        indxsampwalkkeep = np.linspace(numbsampburnwalk, numbsampwalk - 1, numbkeep, dtype=int)
        listparafitt = listparafittwalk[:, indxsampwalkkeep, :].reshape((-1, numbpara))
        
        listparaderi = None
        dictparaderi = dict()
        if retr_dictderi is not None:
            listdictparaderi = [[] for n in indxsamp]
            listdictvarbderi = [[] for n in indxsamp]
            for n in indxsamp:
                listdictparaderi[n], listdictvarbderi[n] = retr_dictderi(listparafitt[n, :], gdat)

            for strg, valu in listdictparaderi[0].items():
                dictparaderi[strg] = np.empty([numbsamp] + list(valu.shape))
                for n in indxsamp:
                    dictparaderi[strg][n, ...] = listdictparaderi[n][strg]
            numbparaderi = len(listdictparaderi[0])
            listparaderi = np.empty((numbsamp, numbparaderi)) 
            k = 0
            for strg, valu in listdictparaderi[0].items():
                listparaderi[:, k] = dictparaderi[strg][:, 0]
                k += 1
                for n in indxsamp:
                    dictparaderi[strg][n, ...] = listdictparaderi[n][strg]
            
        indxsampwalk = np.arange(numbsampwalk)
        
        if pathimag is not None:
            # plot the posterior
            ### trace
            figr, axis = plt.subplots(numbpara + 1, 1, figsize=(12, (numbpara + 1) * 4))
            for i in indxwalk:
                axis[0].plot(indxsampwalk, listlposwalk[i, :])
            axis[0].axvline(numbsampburnwalk, color='k')
            axis[0].set_ylabel('log P')
            for k in indxpara:
                for i in indxwalk:
                    axis[k+1].plot(indxsampwalk, listparafittwalk[i, :, k])
                labl = listlablpara[k][0]
                if listlablpara[k][1] != '':
                    labl += ' [%s]' % listlablpara[k][1]
                axis[k+1].axvline(numbsampburnwalk, color='k')
                axis[k+1].set_ylabel(labl)
            path = pathimag + 'trac%s.%s' % (strgextn, typefileplot)
            if verbtype == 1:
                print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            # plot the posterior
            ### trace
            if numbsampburnwalk > 0:
                figr, axis = plt.subplots(numbpara + 1, 1, figsize=(12, (numbpara + 1) * 4))
                for i in indxwalk:
                    axis[0].plot(indxsampwalk[numbsampburnwalk:], listlposwalk[i, numbsampburnwalk:])
                axis[0].set_ylabel('log P')
                for k in indxpara:
                    for i in indxwalk:
                        axis[k+1].plot(indxsampwalk[numbsampburnwalk:], listparafittwalk[i, numbsampburnwalk:, k])
                    labl = listlablpara[k][0]
                    if listlablpara[k][1] != '':
                        labl += ' [%s]' % listlablpara[k][1]
                    axis[k+1].set_ylabel(labl)
                path = pathimag + 'tracgood%s.%s' % (strgextn, typefileplot)
                if verbtype == 1:
                    print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
    
    if pathimag is not None:
        ## joint PDF
        strgplot = 'postparafitt' + strgextn
        plot_grid(pathimag, strgplot, listparafitt, listlablpara, numbbinsplot=numbbins)
        
        # derived
        if retr_dictderi is not None:
            listlablparatotl = listlablpara + listlablparaderi
            listparatotl = np.concatenate([listparafitt, listparaderi], 1)
            strgplot = 'postparaderi' + strgextn
            plot_grid(pathimag, strgplot, listparaderi, listlablparaderi, numbbinsplot=numbbins)
            strgplot = 'postparatotl' + strgextn
            plot_grid(pathimag, strgplot, listparatotl, listlablparatotl, numbbinsplot=numbbins)
    
    if strgsaveextn is not None:
        print('Writing to the initial state from %s...' % strgsaveextn)
        np.savetxt(strgsaveextn, np.median(listparafitt, 0))
    
    dictparafitt = dict()
    for k, name in enumerate(listnamepara):
        dictparafitt[name] = listparafitt[:, k]
    dictparafitt['lpos'] = listlposwalk.flatten()
    
    return dictparafitt, dictparaderi

