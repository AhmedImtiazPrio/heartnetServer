import matlab.engine
import numpy as np
from heartnet_v1 import heartnet
from heartnetServer_test import preprocessing, matlab_init, segmentation
import argparse
from scipy.io.wavfile import read

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='Specify wav file to process')
    parser.add_argument("--file",
                        help="path to PCG .wav file")
    args = parser.parse_args()
    if args.file:
        wavfile = args.file
    else:
        wavfile = 'test.wav_'
    print ("Evaluating %s " % (wavfile))

    load_path='weights.0148-0.8902.hdf5'
    target_fs=1000
    nsamp=2500
    matfunctions = 'matfunctions/'

    eng = matlab_init() # Init Matlab
    model = heartnet(load_path) # Build Heartnet Model
    in_fs,PCG = read(wavfile)
    PCG = matlab.double([np.ndarray.tolist(PCG)])  ## Typecast for matlab
    PCG = preprocessing(PCG=PCG, eng=eng, in_fs=in_fs, target_fs=target_fs)
    x = segmentation(PCG=PCG, eng=eng, nsamp=nsamp, target_fs=target_fs)
    y_pred = model.predict(x,verbose=1)

    print("Abnormal probability %f" % np.mean(y_pred))
    if np.mean(y_pred) > .5:
        print("Abnormal")
    else:
        print("Normal")
