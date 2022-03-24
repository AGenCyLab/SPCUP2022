import librosa
import numpy as np
import librosa.filters
import scipy as sc

def cqcc(x, fs, B, fmax, fmin, d, cf, ZsdD):
    """Constant Q cepstral coefficients

    Parameters
    ----------
    x        : input signal
    fs       : sampling frequency
    B        : number of bins per octave [default = 96]
    fmax     : highest frequency to be analyzed [default = Nyquist frequency]
    fmin     : lowest frequency to be analyzed [default = ~20Hz to fullfill an integer number of octave]
    d        : number of uniform samples in the first octave [default 16]
    cf       : number of cepstral coefficients excluding 0'th coefficient [default 19]
    ZsdD     : any sensible combination of the following  [default ZsdD]:
                      'Z'  include 0'th order cepstral coefficient
                      's'  include static coefficients (c)
                      'd'  include delta coefficients (dc/dt)
                      'D'  include delta-delta coefficients (d^2c/dt^2)

    see : 
    Returns
    -------
    cqcc_feature : np.ndarray [shape=(n,) or (..., n)]
        features
    """

    gamma = 228.7 * (2**(1/B) - 2**(-1/B) )
    eps = np.finfo(float).eps
    
    # computing cqt
    Xcq = librosa.cqt(x, 
                      sr=fs,
                      hop_length=512,
                      bins_per_octave=B, 
                      fmin=fmin,
                      )
    
    # log power spectrum
    absCQT = np.abs(Xcq)
    timeVec = np.arange(Xcq.shape[1])*x.shape[0]/Xcq.shape[1]/fs
    FreqVec = fmin*(2**(np.arange(Xcq.shape[0]))/B)
    logP_absCQT = np.log(absCQT**2 + eps)

    # uniform resampling
    k1 = (B*np.log2(1+1/d))
    # downsampling
    Ures_logP_absCQT = librosa.resample(logP_absCQT, fs, 9_562)
    #CQceptrum = librosa.filters.dct(Ures_logP_absCQT)
    CQceptrum = sc.fftpack.dct(Ures_logP_absCQT)
    cqcc_feature = CQceptrum[0:cf]

    return cqcc_feature