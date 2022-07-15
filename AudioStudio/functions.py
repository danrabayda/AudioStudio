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
def norm_seq_to_spg(s,r_smp,v_res,f_len,t_len,filt):
    s = normalize_seq(s) #first normalize the window
    sp = sequence_to_spectrogram(s,r_smp,v_res,f_len,t_len) - np.repeat(filt[:,np.newaxis],t_len,axis=1)
    return normalize_spectrogram(sp)
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
