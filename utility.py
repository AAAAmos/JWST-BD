import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from scipy.optimize import curve_fit
from scipy.interpolate import griddata
from astropy.coordinates import SkyCoord
from astropy import units as u
from astropy.table import Table
from astropy.coordinates import match_coordinates_sky

import lmfit
from lmfit.lineshapes import gaussian2d
from lmfit.models import LorentzianModel
from lmfit.models import ExpressionModel

def gaussian(x, amp, cen, sig):
    """
    1-d gaussian: gaussian(x, amp, cen, wid)
    
    input: number, array
    """
    return (amp / (np.sqrt(2*np.pi) * sig)) * np.exp(-(x-cen)**2 / (2*sig**2))

def gaussian2d(x, y, amp, cenx, ceny, sigx, sigy):
    """
    2-d gaussian: gaussian(x, y, amp, cenx, ceny, sigx, sigy)
    
    input: number, array
    """
    return (amp / (2*np.pi * sigx*sigy)) * np.exp(-((x-cenx)**2 / (2*sigx**2)) - ((y-ceny)**2 / (2*sigy**2)))

def Poisson(x, k):
    
    return (np.exp(-k)*k**x)/np.math.factorial(x)

def CrossmatchDisfit(file, cname, cname0=['RA','DEC'], fitrange=70, grid=101, weight=1, mode=2):
    
    """
    This function is used to fit the crossmatch distance distribution of two catalogs.
    
    Return: 2D fitting plot
    
    Parameters:
    file: str, the file path of the crossmatch result
    cname: list, the column name of the second catalogs, first catalog is RA and DEC
    fitrange: int, the range of the fitting plot in arcsec
    grid: int, the number of the grid in the fitting plot, need to be odd
    weight: float, the weight of the fitting to the data
    mode: int, the mode of the fitting
        1: 1D fitting
        2: 2D fitting with ra, dec as the x, y axis
        3: 2D fitting with only radius variable
    """

    if type(file) == str:
        data = pd.read_csv(file)

        n1 = cname[0]
        n2 = cname[1]
    else:
        data = file
        n1 = cname[0]
        n2 = cname[1]

    x = (data[cname0[0]]-data[n1])*3600*np.cos(data[cname0[1]]*np.pi/180)
    y = (data[cname0[1]]-data[n2])*3600
    
    if mode == 1:
        
        fig, ax = plt.subplots()
        
        data = {
            'ra': ax.hist(x, grid)[1],
            'rac': ax.hist(x, grid+1)[0],
            'dec': ax.hist(y, grid)[1],
            'decc': ax.hist(y, grid+1)[0]
        }
        
        plt.close(fig)
        
        xedges = np.linspace(-fitrange, fitrange, grid)
        yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()
        
        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf = X.flatten()
        yf = Y.flatten()
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        vmax = np.nanpercentile(Z, 99.9)

        dframe = pd.DataFrame(data=data)

        model = LorentzianModel()

        paramsx = model.guess(dframe['rac'], x=dframe['ra'])
        paramsy = model.guess(dframe['decc'], x=dframe['dec'])

        resultra = model.fit(dframe['rac'], paramsx, x=dframe['ra'])
        cen1x = resultra.values['center']
        sig1x = resultra.values['sigma']
        resultdec = model.fit(dframe['decc'], paramsy, x=dframe['dec'])
        cen1y = resultdec.values['center']
        sig1y = resultdec.values['sigma']
        
        fitx = model.func(dframe['ra'], **resultra.best_values)
        fity = model.func(dframe['dec'], **resultdec.best_values)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        
        plt.rcParams.update({'font.size': 15})
        # ax = axs[0]
        # art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        # plt.colorbar(art, ax=ax, label='z')
        # ell = Ellipse(
        #         (cen1x, cen1y),
        #         width = 3*sig1x,
        #         height = 3*sig1y,
        #         edgecolor = 'w',
        #         facecolor = 'none'
        #     )
        # ax.add_patch(ell)
        # ax.set_title('Histogram of Data')
        # ax.set_xlabel('Delta RA [arcsec]')
        # ax.set_ylabel('Delta DEC [arcsec]')

        ax = axs[0]
        ax.plot(dframe['ra'], fitx, label='fit gaussian')
        ax.plot(dframe['ra'], dframe['rac'], 
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('Center:{0:5.4f}, 1 Sigma:{1:5.3f}'.format(cen1x, sig1x))
        ax.set_xlabel('Delta RA [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()

        ax = axs[1]
        ax.plot(dframe['dec'], fity, label='fit gaussian')
        ax.plot(dframe['dec'], dframe['decc'], 
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('Center:{0:5.4f}, 1 Sigma:{1:5.3f}'.format(cen1y, sig1y))
        ax.set_xlabel('Delta DEC [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()
        
        fig.suptitle('AKARI-TSL3 x '+file.split('-')[1][:-6] + '  1D fitting')
        
        plt.show()
    
    if mode == 2:

        xedges = np.linspace(-fitrange, fitrange, grid)
        yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()

        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf = X.flatten()
        yf = Y.flatten()
        
        w = z**weight+0.1
        
        model = Gaussian2dModel()
        params = model.guess(z, xf, yf)
        result = model.fit(z, x=xf, y=yf, params=params, weights=w/10)
        Amp = result.values['amplitude']
        cenx = result.values['centerx']
        sigx = result.values['sigmax']
        ceny = result.values['centery']
        sigy = result.values['sigmay']
        
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        vmax = np.nanpercentile(Z, 99.9)
        
        fit = model.func(X, Y, **result.best_values)

        Zx = Z[int((grid+1)/2)]
        fitx = fit[int((grid+1)/2)]
        Zy = Z.T[int((grid+1)/2)]
        fity = fit.T[int((grid+1)/2)]

        fig, axs = plt.subplots(2, 2, figsize=(15, 13))
        
        plt.rcParams.update({'font.size': 15})
        # plt.rcParams.update({"tick.labelsize": 13})
        
        ax = axs[0, 0]
        art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='Data point Density')
        ell = Ellipse(
                (cenx, ceny),
                width = 3*sigx,
                height = 3*sigy,
                edgecolor = 'w',
                facecolor = 'none'
            )
        ax.add_patch(ell)
        ax.set_title('Histogram of Data')
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        ax = axs[0, 1]
        art = ax.pcolor(X, Y, Z-fit, shading='auto')
        plt.colorbar(art, ax=ax, label='Data point Density')
        ax.set_title('Residual')
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)

        ax = axs[1, 0]
        ax.plot(xedges[:grid-1], fitx, label='fit gaussian')
        ax.plot(xedges[:grid-1], Zx, 
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('y-axis slice, Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenx, sigx))
        ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()

        ax = axs[1, 1]
        ax.plot(yedges[:grid-1], fity, label='fit gaussian')
        ax.plot(yedges[:grid-1], Zy,
                marker='s', markersize=5, ls='', label='data point'
                )
        ax.set_title('x-axis slice, Center:{0:5.3f}, 1σ:{1:5.2f}'.format(ceny, sigy))
        ax.set_xlabel('ΔDEC [arcsec]', fontsize=15)
        ax.set_ylabel('count', fontsize=15)
        ax.tick_params(axis='both', labelsize=13)
        ax.legend()

        fig.suptitle('AKARI-TSL3 x '+file.split('-')[1][:-4]+'  2D fitting')

        plt.show()
        
    if mode == 3:

        xedges = yedges = np.linspace(-fitrange, fitrange, grid)
        H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
        H = H.T
        z = H.flatten()

        X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
        xf, yf = X.flatten(), Y.flatten()
        
        model = ExpressionModel(
            'amp*exp(-(x**2 / (2*sig**2)) - (y**2 / (2*sig**2)))',
            independent_vars=['x', 'y']
        )
        params = model.make_params(amp=100, sig=fitrange/100)
        
        w = z**1.1
        result = model.fit(z, x=xf, y=yf, params=params, weights=w)
        Sigma = result.params['sig'].value
        print(Sigma)
        
        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
        Zx = Z[int((grid+1)/2)]
        Zy = Z.T[int((grid+1)/2)]

        fig, axs=plt.subplots(1, 2, figsize=(15, 5), dpi=100)

        ax = axs[0]
        ax.plot(xedges[:grid-1], Zx, 
            marker='s', markersize=5, ls='', label='data points'
            )
        ax.plot(np.linspace(-fitrange, fitrange, 100),
            model.eval(result.params, x=np.linspace(-fitrange, fitrange, 100), y=0),
            label=f'fit gaussian, $\\sigma$={Sigma:.4f}')
        ax.set_title('y-axis slice')
        ax.set_xlabel('Separation [arcsec]')
        ax.legend()

        ax=axs[1]
        ax.plot(yedges[:grid-1], Zy, 
            marker='s', markersize=5, ls='', label='data points'
            )
        ax.plot(np.linspace(-fitrange, fitrange, 100),
            model.eval(result.params, x=0, y=np.linspace(-fitrange, fitrange, 100)),
            label=f'fit gaussian, $\\sigma$={Sigma:.4f}')
        ax.set_title('x-axis slice')
        ax.set_xlabel('Separation [arcsec]')
        ax.legend()
        
        plt.show()
        
        return Sigma

def CrossGaussian2dfit(file, cname, fitrange=60, grid=100, weight=0.9, fitonly=False, sqslice=False):

    data = pd.read_csv(file)

    n1 = cname[0]
    n2 = cname[1]

    x = (data.RA-data[n1])
    y = (data.DEC-data[n2])
    
    if sqslice:
        data = data.loc[(x>-0.02)&(x<0.02)&(y>-0.02)&(y<0.02),
                    ['RA', 'DEC', n1, n2]
                    ]

    x = (data.RA-data[n1])*3600*np.cos(data.DEC*np.pi/180)
    y = (data.DEC-data[n2])*3600

    xedges = np.linspace(-fitrange, fitrange, grid)
    yedges = np.linspace(-fitrange, fitrange, grid)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    # Histogram does not follow Cartesian convention (see Notes),
    # therefore transpose H for visualization purposes.
    H = H.T

    z = H.flatten()

    X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
    xf = X.flatten()
    yf = Y.flatten()
    # error = np.sqrt(z+1)
    w = z**weight+0.1

    model = lmfit.models.Gaussian2dModel()
    params = model.guess(z, xf, yf)
    result = model.fit(z, x=xf, y=yf, params=params, weights=w/10)

    if not fitonly:
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))

        Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)

        vmax = np.nanpercentile(Z, 99.9)

        ax = axs[0, 0]
        art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='z')
        ax.set_title('Histogram of Data')

        ax = axs[0, 1]
        fit = model.func(X, Y, **result.best_values)
        art = ax.pcolor(X, Y, fit, vmin=0, vmax=vmax, shading='auto')
        plt.colorbar(art, ax=ax, label='z')
        ax.set_title('Fit')

        ax = axs[1, 0]
        fit = model.func(X, Y, **result.best_values)
        art = ax.pcolor(X, Y, Z-fit, shading='auto')
        plt.colorbar(art, ax=ax, label='z')
        ax.set_title('Data - Fit')

        ax = axs[1, 1]
        ax.scatter(
            x, y,
            s = 1,
            alpha=0.2
        )
        ax.set_title('Origin data points from'+file.split('-')[1])

        for ax in axs.ravel():
            ax.set_xlabel('Delta RA [arcsec]')
            ax.set_ylabel('Delta DEC [arcsec]')
        
        plt.show()

    if fitonly:
        
        return result

def CrossmatchDisfit2G(file, cname, fitrange=70, grid=101, weight=1, dim=2):

    data = pd.read_csv(file)

    n1 = cname[0]
    n2 = cname[1]

    x = (data.RA-data[n1])
    y = (data.DEC-data[n2])

    x = (data.RA-data[n1])*3600*np.cos(data.DEC*np.pi/180)
    y = (data.DEC-data[n2])*3600

    xedges = np.linspace(-fitrange, fitrange, grid)
    yedges = np.linspace(-fitrange, fitrange, grid)
    H, xedges, yedges = np.histogram2d(x, y, bins=(xedges, yedges))
    H = H.T
    z = H.flatten()

    X, Y = np.meshgrid(np.linspace(-fitrange, fitrange, grid-1), np.linspace(-fitrange, fitrange, grid-1))
    xf = X.flatten()
    yf = Y.flatten()

    w = z**weight+0.1

    model = (lmfit.models.Gaussian2dModel(prefix='g1_')
            +lmfit.models.Gaussian2dModel(prefix='g2_')
            )
    params = model.make_params(
        g1_amplitude=1000,
        g1_center=0,
        g1_sigma=3,
        g2_amplitude=0,
        g2_center=0,
        g2_sigma=20
        )
    result = model.fit(z, x=xf, y=yf, params=params, weights=w/10)
      
    Amp1 = result.best_values['g1_amplitude']
    cenx1 = result.best_values['g1_centerx']
    sigx1 = result.best_values['g1_sigmax']
    ceny1 = result.best_values['g1_centery']
    sigy1 = result.best_values['g1_sigmay']
    
    Amp2 = result.best_values['g2_amplitude']
    cenx2 = result.best_values['g2_centerx']
    sigx2 = result.best_values['g2_sigmax']
    ceny2 = result.best_values['g2_centery']
    sigy2 = result.best_values['g2_sigmay']
    
    if (sigx1 < sigx2) and (sigx1 > 2):
        Ampm = Amp1
        cenxm = cenx1
        cenym = ceny1
        sigxm = sigx1
        sigym = sigy1
        Ampe = Amp2
        cenxe = cenx2
        cenye = ceny2
        sigxe = sigx2
        sigye = sigy2
        
    else:
        Ampm = Amp2
        cenxm = cenx2
        cenym = ceny2
        sigxm = sigx2
        sigym = sigy2
        Ampe = Amp1
        cenxe = cenx1
        cenye = ceny1
        sigxe = sigx1
        sigye = sigy1

    # print(cenxm, sigxm, cenxe, sigxe)
    
    Z = griddata((xf, yf), z, (X, Y), method='linear', fill_value=0)
    vmax = np.nanpercentile(Z, 99.9)

    fit = model.func(X, Y, **result.best_values)

    Zx = Z[int((grid+1)/2)]
    Zy = Z.T[int((grid+1)/2)]
    
    edges = np.linspace(-fitrange, fitrange, 5*grid)
    fitxm = gaussian2d(edges, 0, Ampm, cenxm, cenym, sigxm, sigym) 
    fitym = gaussian2d(edges, 0, Ampm, cenxm, cenym, sigxm, sigym)
    fitxe = gaussian2d(0, edges, Ampe, cenxe, cenye, sigxe, sigye)
    fitye = gaussian2d(0, edges, Ampe, cenxe, cenye, sigxe, sigye)

    fig, axs = plt.subplots(2, 2, layout="constrained", figsize=(14, 10))

    plt.rcParams.update({'font.size': 15})
    # plt.rcParams.update({"tick.labelsize": 13})

    ax = axs[0, 0]
    art = ax.pcolor(X, Y, Z, vmin=0, vmax=vmax, shading='auto')
    plt.colorbar(art, ax=ax, label='Data point Density')
    ell = Ellipse(
            (cenxm, cenym),
            width = 3*sigxm,
            height = 3*sigym,
            edgecolor = 'w',
            facecolor = 'none'
        )
    ax.add_patch(ell)
    ax.set_title('Histogram of Data')
    ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
    ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)

    # ax = axs[0, 1]
    # art = ax.pcolor(X, Y, Z-fit, shading='auto')
    # plt.colorbar(art, ax=ax, label='Data point Density')
    # ax.set_title('Residual')
    # ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
    # ax.set_ylabel('ΔDEC [arcsec]', fontsize=15)
    # ax.tick_params(axis='both', labelsize=13)

    ax = axs[1, 0]
    ax.plot(edges, fitxm, label='matched', lw=3, ls='--')
    ax.plot(edges, fitxe, label='fake-matched?')
    ax.plot(edges, fitxm+fitxe, label='total')
    ax.plot(xedges[:grid-1], Zx, marker='s', markersize=5, ls='', label='data point')
    ax.set_title(
        'y-axis slice, \n Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenxm, sigxm) + 
        ', Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenxe, sigxe)
    )
    ax.set_xlabel('ΔRA [arcsec]', fontsize=15)
    ax.set_ylabel('count', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend()

    ax = axs[1, 1]
    ax.plot(edges, fitym, label='matched', lw=3, ls='--')
    ax.plot(edges, fitye, label='fake-matched?')
    ax.plot(edges, fitym+fitye, label='total')
    ax.plot(yedges[:grid-1], Zy, marker='s', markersize=5, ls='', label='data point')
    ax.set_title(
        'x-axis slice, \n Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenym, sigym) + 
        ', Center:{0:5.3f}, 1σ:{1:5.2f}'.format(cenye, sigye)
    )
    ax.set_xlabel('ΔDEC [arcsec]', fontsize=15)
    ax.set_ylabel('count', fontsize=15)
    ax.tick_params(axis='both', labelsize=13)
    ax.legend()

    fig.suptitle('AKARI-TSL3 x '+file.split('-')[1][:-6]+'  2D fitting')

    plt.show()

def Errorbar(x, y, yerr, c='black', elinewidth=1, alpha=1):
    
    '''
    The plt.errorbar has problem on the oerder of layer, so I am making this function
    x: np.array
    y: np.array
    yerr: np.array
    
    This is a garbige, exaggerate zorder
    '''
    
    # fig, ax = plt.subplots()
    
    for i in range(len(x)):
        
        ymax1 = y[i]+yerr[i]
        ymin1 = y[i]-yerr[i]
        # print(ymax1, ymin1)
        plt.vlines(x=x[i], ymin=ymin1, ymax=ymax1, colors=c, linewidth=elinewidth, alpha=alpha)
        
def crossmatch(data1, data2, radius=1, offset=False):
    
    ''' Crossmatch two datasets using astropy.coordinates.match_coordinates_sky
    and return a merged dataset with matched rows from both datasets.
    
    Parameters:
    data1 (DataFrame): First dataset to crossmatch
    data2 (DataFrame): Second dataset to crossmatch
    radius (float): Maximum separation between matches in arcseconds
    
    Returns:
    merged_data (DataFrame): Merged dataset containing matched rows from data1 and data2
    
    ''' 
    
    # Create SkyCoord objects for both datasets
    coords1 = SkyCoord(ra=data1['ALPHA_J2000'], dec=data1['DELTA_J2000'], unit="deg")
    coords2 = SkyCoord(ra=data2['ALPHA_J2000'], dec=data2['DELTA_J2000'], unit="deg")
    
    if offset:
        # Define the region with offset
        region_mask = (
            (data2['DELTA_J2000'] < 2.79 * (data2['ALPHA_J2000'] - 150.18) + 2.0922) &
            (data2['DELTA_J2000'] > 2.79 * (data2['ALPHA_J2000'] - 150.305) + 2.0482) &
            (data2['DELTA_J2000'] > -0.352 * (data2['ALPHA_J2000'] - 150.18) + 2.0922)
        )
        
        # Apply offset to coords1 for sources in the specified region, for 277 band, A4
        coords2[region_mask] = SkyCoord(
            ra=coords2.ra[region_mask] + 0.18*u.arcsec,
            dec=coords2.dec[region_mask] - 0.06*u.arcsec,
            unit="deg"
        )

    # Find the nearest neighbors in coords2 for each point in coords1
    idx, d2d, d3d = match_coordinates_sky(coords1, coords2, nthneighbor=1)

    # Create a mask for matches within the specified radius
    mask = d2d.arcsec <= radius

    # Add matching indices and distances to data1
    data1['idx'] = idx
    data1['d2d'] = d2d.arcsec

    # Filter data1 and data2 to only include matches, iloc is used to keep the same index
    matched_data1 = data1[mask].reset_index(drop=True)
    matched_data2 = data2.iloc[idx[mask]].reset_index(drop=True)

    # Merge matched rows from data1 and data2
    merged_data = pd.concat([matched_data1, matched_data2], axis=1)

    return merged_data