# AudioStudio
### A pip package that makes quick work of converting time-series data into spectrograms as well as playable audio


Here is a list of the functions as well as explanations for each in a shortened list:

flat(bl)
flattens a list of lists into just a list

i_str(i)
makes 0,1,2,...13,14,etc into 00,01,02,...,13,14,etc for file saving purposes

reduce_seq(s,xlr)
takes in a sequence and reduces it to a 1d list only if it is 2d and based on the LRB parameter

normalize_seq(s)
normalizes sequence s to 0 to 1

xlr(lrb)
used to read the left or right audio parameter quickly, options for lrb are "L", "R", and any other string like "B" which will default to both channels

pydub_to_np(audio, r_smp=44100)
Converts pydub audio segment into np.float32 of shape [duration_in_seconds*sample_rate, channels], where each value is in range [-1.0, 1.0]. Returns audio_np_array

sequence_to_spectrogram(s,r_smp,v_res,f_len,t_len)
takes in a sequence (1 ch) and ouputs it as a log spectrogram in greyscale (1 ch)

norm_seq_to_spg(s,r_smp,v_res,f_len,t_len,filt)
takes in raw sample data s, normalizes it, creates a spectrogram from it, then returns a normalized spectrogram

normalize_spectrogram(sp)
normalizes a spectrogram sp to a range 0 to 1

quick_spectrogram(s,r_smp=44100,v_res=2**9)
uses default values for everything so you only need to give it a segment of samples

quick_sound(s,r_smp=44100)
make an Audiosegment from a sequence

quick_plot(sp)
make a quick plot using default values

quick_example(s,r_smp=44100)
does a quickplot for sequence s and shows a play and save button to play or save the audio of that sequence

decimate(s,new_rate,old_rate=44100)
rough downsampling from one freq to a new lower one, I made this becasue scipy.signal.decimate only does integer downsampling, mine is general

lengthwise_median_filter(sp,res,stride=1)
median filter in only the lengthwise direction for a greyscale spectrogram. the resolution is the number of pixels lengthwise to include in the filter.

multiple_filter(funcs,params,ai)
iterates one function f at a time to make f(f(f(a,pi))), where params=[p1,p2,p3]. The number of iterations equal to the length of params & funcs arrays. Available funcs are 'm' (median filter), 'g' (gaussian filter), and 'l' (lengthwise median filter), the params for these specifically are integer filter resolutions.

vdir(directory)
verify a directory exists, if not make it

vdirs(directory1,directory2)
just does vdir(os.path.join(directory1,directory2))


### 0.0.3:

vdir_batch(dirs): #batch creates dirs in the current directory and returns a tuple of strings representing their locations

vdirs_batch(parent_dir,subdirs): #batch creates dirs in the parent directory and returns a tuple of strings representing their locations

dropout_filter(sp,thr=0.75): #a filter that drops any pixel values below the threshold thr down to 0, e.g. [0,0.3,0.7,0.8,0.3] would become [0,0,0,0.8,0] for thr=0.75

get_spectrogram_dims(pdict): #returns the default spectrogram dims for a given pdict using scipy's values unless the dims are resized using pdict['rd']

attempt_instantiation(pdict,param_str,except_val): #attempts instantiating param = pdict[param_str] and if there is no pdict entry it sets param=except_val

average_frequency_band_filter(data, pdict): #creates a custom bandpass filter using a spectrogram of the entire data file data_array[dind]=data

scale_arr(arr,sc): #same as decimate but with a range of sc=0.0001 to 1

place_arr(arr,new_size,spos,fill): #places arr inside a new_arr with length new_size>old_size at position spos (0 to 1)

scpl_sp(sp,scales,places): #scale and place spectrogram columns according to set parameters

random_augmenter(sp,deg,sig_prob=0.5): #automatically augments spectrogram sp using sigmoid, linear, expansion and contraction with a degree from 0 to 1 and a probability of sigmoid placements as sig_prob

get_lengths_under_ws(lengths,r_smp,w): #grabs a reverse list of length indices for lengths less than w seconds

pop_lengths_under_ws(metadata,r_smp,w): #pops lengths under w seconds

labels_to_labels_c(labels,classes): #converts raw labels into indices referencing classes

class_grouper(labels_c,times,n_classes): #returns class-grouped metadata (metadata grouped by class). e.g. class_groups[3] contains all the [file, [time]] references for class 3

cg_splitter(class_groups,split): #splits class_groups into train, validation, and test sets

equalize_class_groups(class_groups): #Our generator picks from all available classes with equal probability, so this is a bit unnecessary, only needed if you want to do a final test or validate against research data (like the drone data)

genC_plot(nP,nK,spectrograms,ls,classes,sps=[],files_as_labels=True): #plots generatorC's output
