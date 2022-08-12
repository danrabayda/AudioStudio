import numpy as np
from pydub import AudioSegment
import os
import io
import scipy.io.wavfile
from scipy import signal, ndimage
import matplotlib.pyplot as plt
from pydub.playback import play
import ipywidgets as widgets
from IPython.display import display, HTML
from IPython.display import Image

def flat(bl):
    return [v for l in bl for v in l] #flattens a list of lists into just a list
def i_str(i):
    return '0'+str(i) if i<10 else str(i) #makes 0,1,2,...13,14,etc into 00,01,02,...,13,14,etc. I use it for file saving purposes
def reduce_seq(s,xlr):
    return s if len(np.shape(s))==1 else s[:,xlr] #takes in a sequence and reduces it to a 1d list only if it is 2d and based on the LRB parameter
def normalize_seq(s):
    return s/(np.max(np.abs(s))+1e-8) #normalizes sequence s
def xlr(lrb):
    return 0 if lrb=='L' else 1 if lrb=='R' else np.random.randint(2) #used to read the left or right audio parameter quickly
def meme(lnk,txt): #displays a hyperlink to a meme with text txt
    display(HTML("""<a href={link}>{text}</a>""".format(link=lnk,text=txt)))
def pydub_to_np(audio, r_smp=44100): #Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels], where each value is in range [-1.0, 1.0]. Returns audio_np_array
    audio.set_frame_rate(r_smp)
    return np.array(audio.get_array_of_samples(), dtype=np.float32).reshape((-1, audio.channels)) / (1 << (8 * audio.sample_width - 1))
def sequence_to_spectrogram(s,r_smp,v_res,f_len,t_len): #takes in a sequence (1 ch) and ouputs it as a log spectrogram in greyscale (1 ch)
    _, _, spectrogramx = signal.spectrogram(s,r_smp,nperseg=v_res)
    log_spectrogram=np.log10(spectrogramx[:f_len,:t_len],out=spectrogramx[:f_len,:t_len],where=spectrogramx[:f_len,:t_len] > 0)
    diff=np.max(log_spectrogram)-np.min(log_spectrogram)
    spectrogram=(log_spectrogram-np.min(log_spectrogram))/diff if diff!=0 else log_spectrogram-np.min(log_spectrogram)
    return spectrogram
def normalize_spectrogram(sp):
    sp = sp - np.min(sp)
    sp = sp/(np.max(sp)+1e-8)
    return sp
def quick_spectrogram(s,r_smp=44100,v_res=2**9): #uses default values for everything so you only need to give it a segment of samples
    t_len=int((len(s)-(v_res/8))/(v_res*7/8))
    f_len=int(.75 * v_res/2)+1 #70
    return sequence_to_spectrogram(s,r_smp,v_res,f_len,t_len)
def quick_sound(s,r_smp=44100): #similarly make a sound from a segment using defaults
    wav_io = io.BytesIO()
    scipy.io.wavfile.write(wav_io,r_smp, s)     #creates a quick audio snippet out of the segment
    wav_io.seek(0)
    return AudioSegment.from_wav(wav_io)
def quick_plot(sp): #make a quick plot using default values
    plt.figure(figsize=(20, 5))
    plt.imshow(sp,interpolation='nearest',aspect='auto')
    plt.show()
def quick_example(s,r_smp=44100,export_filename='quick_example.wav'): #does a quickplot and quicksound on sequence s
    quick_plot(quick_spectrogram(s,r_smp))
    def play_sound(arg):
        sound=quick_sound(s,r_smp)
        play(sound)
    def save_sound(arg):
        sound=quick_sound(s,r_smp)
        sound.export(export_filename,format='wav')
        print("saved to "+export_filename)
    play_button=widgets.Button(description='Play Audio')
    play_button.on_click(play_sound)
    display(play_button)
    save_button=widgets.Button(description='Save Audio')
    save_button.on_click(save_sound)
    display(save_button)
def decimate(s,new_rate,old_rate=44100): #rough downsampling from one freq to a new lower one, I made this becasue scipy.signal.decimate only does integer downsampling, mine is general
    skp=old_rate/new_rate
    new_s,sp=[],0 
    for i in range(len(s)):
        if i==round(sp): #I set this to round instead of int.
            sp+=skp
            new_s.append(s[i])
    return np.array(new_s)
def lengthwise_median_filter(sp,res,stride=1): #median filter in only the lengthwise direction for a greyscale spectrogram
    new_sp=np.zeros(sp.shape)
    pw=(res-stride+1)/2 #same padding
    sp=np.pad(sp,((0,0),(int(pw),int(np.ceil(pw)))),'edge')
    for i in range(0,len(sp[0])-res,stride):
        new_sp[:,i]=np.median(sp[:,i:i+res],axis=-1)
    return np.array(new_sp)
def multiple_filter(funcs,params,ai): #iterates a list of single parameter functions on ob with the number of iterations equal to the length of params&or funcs
    func_d={"m":ndimage.median_filter,"l":lengthwise_median_filter,"g":ndimage.gaussian_filter}
    a=ai
    for i in range(len(params)):
        a=func_d[funcs[i]](a,params[i])
    return a
def vdir(directory): #verify a directory exists, if not make it
    if not os.path.exists(directory): os.mkdir(directory)
    return directory
def vdirs(directory1,directory2):
    return vdir(os.path.join(directory1,directory2))


def norm_seq_to_spg(s,r_smp,v_res,f_len,t_len,filt=[]):
    if filt==[]:
        filt=[0 for i in range(f_len)]
    s = normalize_seq(s) #first normalize the window
    sp = sequence_to_spectrogram(s,r_smp,v_res,f_len,t_len) - np.repeat(filt[:,np.newaxis],t_len,axis=1)
    return normalize_spectrogram(sp)

def vdir_batch(dirs):
    return (vdir(d) for d in dirs)
def vdirs_batch(parent_dir,subdirs):
    return (vdirs(parent_dir,subdir) for subdir in subdirs)
def dropout_filter(sp,thr=0.75): #a filter that drops any pixel values below the threshold thr down to 0, e.g. [0,0.3,0.7,0.8,0.3] would become [0,0,0,0.8,0] for thr=0.75
    max_pixel=np.max(sp)
    return sp*(sp>thr*max_pixel)
def get_spectrogram_dims(pdict): #returns the default spectrogram dims for a given pdict using scipy's values unless the dims are resized using pdict['rd']
    height = int((int(pdict['w_sec']*pdict['r_smp'])-(pdict['vres']/8))/(pdict['vres']*7/8))
    width = int(.75 * pdict['vres']/2)+1
    shape=(height,width)
    resize_div=attempt_instantiation(pdict,'rd',0)
    if resize_div>0:
        shape=tuple((np.array(shape)//resize_div)*resize_div)
    return shape
def attempt_instantiation(pdict,param_str,except_val): #attempts instantiating param = pdict[param_str] and if there is no pdict entry it sets param=except_val
    try:
        p=pdict[param_str]
    except:
        p=except_val
    return p
def average_frequency_band_filter(data, pdict): #creates a custom bandpass filter using a spectrogram of the entire data file data_array[dind]=data
    v_res=pdict['vres']
    t_arr=int((len(data)-(v_res/8))/(v_res*7/8)) 
    f_len=int(.75 * v_res/2)+1
    carr=data[:,xlr(pdict['LRB'])]
    total_spec=sequence_to_spectrogram(carr,pdict['r_smp'],v_res,f_len,t_arr)
    mn=np.mean(total_spec,axis=-1)
    return mn

# Data Augmentation Functions
def scale_arr(arr,sc): #same as decimate but with a range of sc=0.0001 to 1
    if sc==0: sc=0.0001 #handles accidental true zero
    return decimate(arr,sc,1)
def place_arr(arr,new_size,spos,fill): #places arr inside a new_arr with length new_size>old_size at position spos (0 to 1)
    pos=int(spos*(new_size-len(arr))) #spos goes from 0 to 1 so we make the true position pos equal to spos times our range of valid positions
    new_arr=np.ones(new_size)*fill
    for i in range(len(arr)):
        new_arr[i+pos]=arr[i]
    return new_arr
def scpl_sp(sp,scales,places): #scale and place spectrogram columns according to set parameters
    spt=sp.T
    fill=np.mean(sp)
    new_spt=np.array([place_arr(scale_arr(spt[i],scales[i]),len(spt[i]),places[i],fill) for i in range(len(scales))])
    return new_spt.T
def random_augmenter(sp,deg,sig_prob=0.5): #automatically augments spectrogram sp using sigmoid, linear, expansion and contraction with a degree from 0 to 1 and a probability of sigmoid placements as sig_prob
    t_res=sp.shape[1]
    def random_sigmoid(ci=0.7): #c can be any number, ideally between 0.5 and 6ish
        c=ci+((np.random.random()*5.5-(ci-0.5))*deg)
        sig=np.array([1/(1+np.exp((-c*np.log(t_res)*(2*x-t_res))/(t_res))) for x in range(t_res)])
        return 1-sig if np.random.randint(2) else sig
    def random_line(mi=0,bi=0.95): #range m=-1 b=1 to m=1 b=0. constraints are b from 0 to 1 and m+b from 0 to 1
        b=bi-((np.random.random()-(1-bi))*deg)
        m=-4
        while m+b>1 or m+b<0:
            m=mi+((np.random.random()*2-(1+mi))*deg)
        return np.array([(m/t_res)*x+b for x in range(t_res)])
    scales=random_line()
    places=random_sigmoid() if np.random.random()<sig_prob else random_line(mi=0,bi=0.5)
    #plt.plot(scales);plt.plot(places);plt.ylabel("placement and scaling");plt.legend(['scales','places']);plt.show()
    return scpl_sp(sp,scales,places) if np.random.random()<0.66 else sp  #make a 1/3 chance of just spitting back the original spectrogram instead of the augmented one

#metadata functions
def get_lengths_under_ws(lengths,r_smp,w): #grabs a reverse list of length indices for lengths less than w seconds
    lengths_under_ws=[i for i,length in enumerate(np.array(lengths)/r_smp) if length<w]
    lengths_under_ws.reverse() #handy for popping out of lists this way
    return lengths_under_ws
def pop_lengths_under_ws(metadata,r_smp,w): #pops lengths under w seconds
    lengths=metadata[-1].tolist()
    metadata=metadata[:-1].T.tolist()
    lengths_under_ws=get_lengths_under_ws(lengths,r_smp,w)
    for i in lengths_under_ws:
        metadata.pop(i); lengths.pop(i)
    return np.array(metadata,dtype='object').T.tolist()+[lengths] #returns everything rejoined how it was
def labels_to_labels_c(labels,classes): #converts raw labels into indices referencing classes
    return [[classes.index(l) for l in labelsi] for labelsi in labels]
def class_grouper(labels_c,times,n_classes): #returns class-grouped metadata (metadata grouped by class). e.g. class_groups[3] contains all the [file, [time]] references for class 3
    class_groups=[] #this entire process can be done in one line using np.squeeze, but it is much harder to interpret in my opinion: #[np.squeeze([[[i,times[i][j]] for j in range(len(labels_c[i])) if labels_c[i][j]==c] for i in range(len(labels_c)) if c in labels_c[i]]).tolist() for c in range(len(classes))]
    for c in range(n_classes):
        tmp_cgroup=[]
        for i in range(len(labels_c)):
            for j in range(len(labels_c[i])):
                if labels_c[i][j]==c:
                    tmp_cgroup.append([i,times[i][j]]) #notice how i is just the reference for our files[i]. This can also be used to reference the actual i'th data array when all the files are loaded in order
        class_groups.append(tmp_cgroup)
    return class_groups
def cg_splitter(class_groups,split): #splits class_groups into train, validation, and test sets
    if len(split)>2: split=split[-2:] #this line incase someone inputs [0.7,0.15,0.15] instead of [0.15,0.15] for example
    trn,val,tst=[],[],[]
    for i in range(len(class_groups)):
        tvn=[int(np.ceil(len(class_groups[i])*spl)) for spl in split]
        ttvn=[len(class_groups[i])-np.sum(tvn)]+tvn
        [a1,a2,a3]=[np.sum(ttvn[:j+1]) for j in range(len(ttvn))]
        trn.append(class_groups[i][:a1])
        val.append(class_groups[i][a1:a2])
        tst.append(class_groups[i][a2:])
    return trn,val,tst
def equalize_class_groups(class_groups): #Our generator picks from all available classes with equal probability, so this is a bit unnecessary, only needed if you want to do a final test or validate against research data (like the drone data)
    lens=[len(g) for g in class_groups]
    l0=np.min(lens)
    rand_inds=[np.sort(np.random.permutation(l)[:l-l0]) for l in lens]
    for j in range(len(rand_inds)):
        for ind in reversed(rand_inds[j]):
            class_groups[j].pop(ind)
def genC_plot(nP,nK,spectrograms,ls,classes,sps=[],files_as_labels=True): #plots generatorC's output
    fig, ax = plt.subplots(nP, nK, figsize=(20,nP*2))
    for p in range(nP):
        for k in range(nK):
            idx = p*nK+k    
            spectrogram = spectrograms[idx]
            label = ls[idx]
            ax[p,k].imshow(spectrogram, interpolation='nearest', aspect='auto')
            ax[p,k].set_xticklabels([])
            ax[p,k].set_yticklabels([])
            if not sps==[]:
                if files_as_labels:
                    ax[p,k].set_xlabel(str(sps[2*idx]))
                else:
                    ax[p,k].set_xlabel(""+str(sps[2*idx])+"    "+str(sps[2*idx+1]//(44100*60)))
        ax[p,0].set_ylabel(classes[label])
