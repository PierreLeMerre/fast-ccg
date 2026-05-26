import numpy as np


def simulate_spikemat(dim= [5, 75, 2005],probthr=0.7):
    #the higher probthr [0,1], the fewer spikes
    spikemat = np.random.rand(*dim)
    spikemat[spikemat > probthr] = 1
    spikemat[spikemat <= probthr] = 0
    return spikemat
def xcorr_unnormed(X, Y, NFFT):
    '''fft-based cross-correlation, works like matlab xcorr with norm: coeff'''

    return np.fft.fftshift(np.fft.ifft(np.fft.fft(X, NFFT) * np.conj(np.fft.fft(Y, NFFT))))


next_pow2 = lambda myint: 2**myint.bit_length() #this is my own fn ;) more elegant than Siegle


def jitter(data_in,l,trialax=0,timeax=1):

    data = data_in.transpose(timeax,trialax) #tpts x trials
    #print(data.shape)

    psth = np.mean(data,axis=1) #tpts : averaged across tirals
    n_tpts_orig,n_trials = data.shape

    if np.mod(n_tpts_orig, l):
        #here we should have an append --> Siegle has a halfhearted, non-working matlab copy here!
        pts_to_add = np.mod(-n_tpts_orig, l)
        data = np.append(data,np.zeros((pts_to_add,n_trials)),axis=0)
        psth = np.append(psth,np.zeros((pts_to_add)),axis=0)

    n_tpts = data.shape[0]
    n_wins = n_tpts//l

    data_snipmat = np.reshape(data, [l, n_wins, n_trials], order='F')  #this is wrong in the Siegle script! l x nwins x ntrials
    dataj = np.squeeze(np.sum(data_snipmat, axis=0))  #wins x trials --> spikesum in each window
    psth_snipmat = np.reshape(psth, [l, n_wins], order='F')  #l x wins
    psthj = np.squeeze(np.sum(psth_snipmat, axis=0))  #wins --> sum the spikes in each window


    psthj = np.reshape(psthj,[n_wins,1], order='F')#nwins x 1 #adding an empty middle axis for the trial dimension I guess
    psthj[psthj==0] = 10e-10

    psthj_tiled = np.tile(psthj,[1, n_trials]) #nwins x ntrials --> same totalcount per win for each trial
    corr = dataj/psthj_tiled#nwins x ntrials --> each trial normalized by overall spike in window count
    corr = np.reshape(corr,[1,n_wins,n_trials], order='F')# 1 x nwins x ntrials --> giving one extra dimension
    corr = np.tile(corr, [l, 1, 1])#l x nwins x ntrials  --> so for each snippt one normalization
    corr = np.reshape(corr,[l*n_wins,n_trials], order='F') #(l x nwins) x ntrials  --> is the exchange happening here?!

    psth = np.reshape(psth,[n_tpts,1], order='F')#one extra middle trial axis for the psth: ntpts x 1
    psth_tiled = np.tile(psth,[1, n_trials]) #ntpts x ntrials
    output = psth_tiled*corr #
    output = output[:n_tpts_orig] #ntpts x ntrial only affects the length if it was appended!
    #print(output.shape)
    return output.transpose(timeax,trialax)


def get_pairswise_ccg_jit(spikes1,spikes2,NFFT,jitterwindow=25):
    #spikes is in trials x tpts
    #returns jitter corrected
    xcorr_temp = xcorr_unnormed(spikes1,spikes2,NFFT)#trials x nfftpts --> that is the original ccg for each trial
    xcorr_trialavg = np.squeeze(np.nanmean(xcorr_temp,axis=0))#nfft

    spikes1_jit = jitter(spikes1,jitterwindow,trialax=0,timeax=1) #trials x tpts
    spikes2_jit = jitter(spikes2,jitterwindow,trialax=0,timeax=1) #trials x tpts
    xcorrj_temp = xcorr_unnormed(spikes1_jit,spikes2_jit,NFFT)#trials x nfft
    xcorrj_trialavg = np.squeeze(np.nanmean(xcorrj_temp, axis=0))#average across trials, see above nfft

    return (xcorr_trialavg - xcorrj_trialavg).T #len(target)==(nfft-pad)


def get_ccgjitter(spikemat, jitterwindow=25,index_behavior='corrected'):
    #spikemat is boolean (spike at timept or not) dim: units x trials x timepoints
    #jitterwindow is in timepoints

    n_units, n_trials, n_tpts = spikemat.shape
    FR = spikemat.sum(-1).mean(axis=1)  #dim: nneurons: average number of spikes per trial for each neuron
    NFFT = int(next_pow2(2 * n_tpts))  #twice the datalen but next power of two of that, so 4096 here

    if index_behavior == 'siegle':
        theta = n_tpts - np.abs(np.arange(-(n_tpts - 1), (n_tpts - 1)))  #triangular function
        t2 = np.arange((-n_tpts + 2), n_tpts)  #the correlation time
        target_inds = (t2 + NFFT / 2).astype(int)  # --> from 50 to 4047 in steps of one, used for indexing to get the non-padded values #todo check whether this is done correctly! --> there is a two point difference.

    elif index_behavior == 'corrected':
        theta = n_tpts - np.abs(np.arange(-(n_tpts - 1), n_tpts))  #triangular function
        t2 = np.arange(-(n_tpts - 1), n_tpts)  #this gives you the peak at 0!
        target_inds = (t2 + NFFT / 2).astype(int)

    else: assert 0, 'unknown index behavior: %s'%index_behavior


    n_pairs = int((n_units * (n_units - 1)) / 2)
    ccg_mat = np.zeros((n_pairs, len(t2)), dtype=complex)  #number of pairs x correlation time
    pair_count = 0
    for uidx1 in np.arange(n_units - 1):
        for uidx2 in np.arange(uidx1 + 1, n_units):
            spikes1 = spikemat[uidx1]  #trials x tpts
            spikes2 = spikemat[uidx2]
            ccg_jit = get_pairswise_ccg_jit(spikes1, spikes2, NFFT, jitterwindow=jitterwindow)  #len(t2) corrpts

            # normalize it
            fr_fac = np.sqrt(FR[uidx1] * FR[uidx2])  # firingrate normalization, just a float here
            ccg_normfac = fr_fac * theta  #target len

            ccg_mat[pair_count] = ccg_jit[target_inds] / ccg_normfac  #target_len to exclude the non-padded ones
            pair_count += 1
    return ccg_mat


def get_pairdict(n_units):
    #easy access to identify which index in the ccg_mat to use to get a specific pair
    pairdict = {}
    pair_count = 0
    for uidx1 in np.arange(n_units - 1):
        for uidx2 in np.arange(uidx1 + 1, n_units):
            pairdict[pair_count] = [uidx1,uidx2]
            pair_count += 1
    return pairdict

def get_pairind(unit1,unit2,n_units):
    assert unit1 != unit2, 'unit1 and 2 are the same'
    return [ii for ii,vals in get_pairdict(n_units).items() if vals[0] in [unit1,unit2] and vals[1] in [unit1,unit2]][0]
def get_corrtvec(n_tpts,alignment='corrected'):

    if alignment == 'siegle':
        tvec = np.arange((-n_tpts + 2), n_tpts)  #the correlation time

    elif alignment == 'corrected':
        tvec = np.arange(-(n_tpts - 1), n_tpts)  #this gives you the peak at 0!

    else: assert 0, 'unknown alignment: %s'%alignment

    return tvec
