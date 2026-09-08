from __init__ import *
from tdpy import mcmc
from tdpy import util

def retr_llik_flag(sampvarb, gdatextr):
        
    xpos = sampvarb[0]
    ypos = sampvarb[1]
    
    llik = sp.interpolate.interp2d(gdatextr.xgrd, gdatextr.ygrd, gdatextr.pdfn)(xpos, ypos)
    sampcalc = [arange(5), arange(6)]

    return llik, sampcalc


def retr_datapara():
    
    numbpara = 2

    datapara = util.gdatstrt()
    datapara.indx = dict()
    datapara.minm = zeros(numbpara)
    datapara.maxm = zeros(numbpara)
    datapara.name = empty(numbpara, dtype=object)
    datapara.scal = empty(numbpara, dtype=object)
    datapara.labl = empty(numbpara, dtype=object)
    datapara.unit = empty(numbpara, dtype=object)
    datapara.true = empty(numbpara, dtype=object)
    datapara.vari = zeros(numbpara)


    datapara.indx['xpos'] = 0
    datapara.indx['ypos'] = 0
    datapara.name[0] = 'xpos'
    datapara.name[1] = 'ypos'
    datapara.minm[:] = 0.
    datapara.maxm[:] = 1.
    datapara.scal[:] = 'self'
    datapara.labl[0] = r'$x$'
    datapara.labl[1] = r'$y$'
    datapara.unit[:] = ''
    datapara.vari[:] = 1e-1
    datapara.true[:] = None
    datapara.strg = datapara.labl + ' ' + datapara.unit
    
    return datapara


def cnfg_flag(flagtype='turk'):

    from scipy import ndimage
    gdat = util.gdatstrt()
    pathimag = os.environ["TDPY_DATA_PATH"] + '/imag/'
    pathimagrtag = pathimag + flagtype + 'flag/'
    path = os.environ["TDPY_DATA_PATH"] + '/data/%sflag.png' % flagtype

    # target PDF
    imag = flipud(sp.ndimage.imread(path))
    rati = float(imag.shape[0]) / imag.shape[1]
    xinp = linspace(0., 1., imag.shape[1])
    yinp = linspace(0., 1., imag.shape[0])
    numbxgrd = 200
    numbygrd = int(200 * rati)
    gdat.xgrd = linspace(0., 1., numbxgrd)
    gdat.ygrd = linspace(0., 1., numbygrd)
    if False:
        imag = sp.ndimage.gaussian_filter(imag, sigma=[xinp.size / 300, yinp.size / 300, 0])
    gdat.pdfn = zeros((numbygrd, numbxgrd, 3))
    for k in range(3):
        gdat.pdfn[:, :, k] = sp.interpolate.interp2d(xinp, yinp, imag[:, :, k])(gdat.xgrd, gdat.ygrd)
    gdat.pdfn = 0.3 * gdat.pdfn[:, :, 0]
    
    figr, axis = plt.subplots()
    axis.imshow(imag, extent=[0., 1., 0., 1.], interpolation='none', aspect=rati)
    plt.savefig(pathimagrtag + 'imag.pdf')
    plt.close(figr) 
    
    figr, axis = plt.subplots()
    imag = axis.imshow(gdat.pdfn, extent=[0., 1., 0., 1.], interpolation='none', aspect=rati)
    plt.colorbar(imag)
    plt.savefig(pathimagrtag + 'targ.pdf')
    plt.close(figr) 
    
    # MCMC setup
    verbtype = 1
    datapara = retr_datapara()
    optiprop = True
    
    # run MCMC
    sampbund = mcmc.init(retr_llik_flag, datapara, numbswep=10000000, factthin=100, gdatextr=gdat, pathimag=pathimagrtag, optiprop=True, verbtype=verbtype, rtag='flag')
    listxpos = sampbund[0][:, 0]
    listypos = sampbund[0][:, 1]
    numbsamp = listxpos.size
    numbfram = 100
    for k in range(numbfram):
        indxsamp = int(numbsamp * float(k) / numbfram)
        figr, axis = plt.subplots()
        axis.scatter(listxpos[:indxsamp], listypos[:indxsamp], s=3)

        axis.set_xlim([0., 1.])
        axis.set_ylim([0., 1.])
        path = pathimagrtag + 'thisimag%04d.pdf' % k
        plt.savefig(path)
        plt.close(figr) 
    
    cmnd = 'convert -delay 20 %sthis*.pdf %spostimag.gif' % (pathimagrtag, pathimagrtag)
    os.system(cmnd)


def retr_llik_gaus(sampvarb, gdat):
        
    xpos = sampvarb[0]
    ypos = sampvarb[1]

    llik = 0.
    for n in range(gdat.numbgaus):
        llik -= 0.5 * ((xpos - gdat.xposgaus[n])**2 + (ypos - gdat.yposgaus[n])**2) / gdat.stdvgaus[n]**2

    return llik, []


def retr_llik_ebox(sampvarb, gdat):
        
    xpos = sampvarb[0]
    ypos = sampvarb[1]
    
    llik = 100. * sin(xpos) * sin(ypos)

    return llik, []


def cnfg_gaus():

    # define the target PDF
    minmxpos = 0.
    maxmxpos = 10. * pi
    minmypos = 0.
    maxmypos = 10. * pi

    xgrd = linspace(minmxpos, maxmxpos, 100)
    ygrd = linspace(minmypos, maxmypos, 100)
    
    gdat = util.gdatstrt()
   
    if False:
        gdat.numbgaus = 3
        gdat.xposgaus = rand(gdat.numbgaus) * maxmxpos
        gdat.yposgaus = rand(gdat.numbgaus) * maxmypos
        gdat.stdvgaus = 0.5 + rand(gdat.numbgaus)
        datapara = util.datapara(2 * gdat.numbgaus)
        for k in range(gdat.numbgaus):
            # construct the parameter object
            datapara.defn_para('xpos%04d' % k, minmxpos, maxmxpos, 'self', r'$x$', '', 0.01, gdat.xposgaus[k])
            datapara.defn_para('ypos%04d' % k, minmypos, maxmypos, 'self', r'$y$', '', 0.01, gdat.yposgaus[k])
            
    datapara = util.datapara(2)
    datapara.defn_para('xpos', minmxpos, maxmxpos, 'self', r'$x$', '', 0.01, None)
    datapara.defn_para('ypos', minmypos, maxmypos, 'self', r'$y$', '', 0.01, None)

    # run MCMC
    sampbund = mcmc.init(retr_llik_ebox, datapara, numbswep=100000, optiprop=True, gdatextr=gdat, numbburn=0, factthin=10, rtag='gausgame', numbbinsplot=100, fracrand=0.1)


if len(sys.argv) > 1:
    func = globals().get(sys.argv[1])
    if func is not None:
        func(*sys.argv[2:])

