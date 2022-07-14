# StarScreamLib
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

vdir(directory)
verify a directory exists, if not make it

vdirs(directory1,directory2)
just does vdir(os.path.join(directory1,directory2))
