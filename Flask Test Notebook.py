
# coding: utf-8

# In[28]:


import matlab.engine
from heartnet_v1 import heartnet
import numpy as np
from keras.backend import cast_to_floatx
# from matplotlib import pyplot as plt
import flask
from scipy.io.wavfile import write
import os
# from scipy.io.wavfile import write
# import io

app=flask.Flask(__name__)
model=None

load_path='weights.0148-0.8902.hdf5'
target_fs=1000
in_fs=4000
nsamp=2500
# In[1]:


def preprocessing(PCG,eng,target_fs,in_fs):
    PCG = eng.resample(PCG,matlab.double([target_fs]),matlab.double([in_fs]))
    PCG = eng.butterworth_low_pass_filter(PCG,matlab.double([2]),matlab.double([400]),matlab.double([1000]))
    PCG = eng.butterworth_high_pass_filter(PCG,matlab.double([2]),matlab.double([25]),matlab.double([1000]))
    PCG = eng.schmidt_spike_removal(PCG,matlab.double([target_fs]))
    return PCG


# In[4]:


def matlab_init():
    eng = matlab.engine.start_matlab()
    # eng.addpath('/media/taufiq/Data/heart_sound/Heart_Sound/codes/cristhian.potes-204/');
    eng.addpath(os.path.join(os.getcwd(),'matfunctions/'))
    return eng


# In[25]:


def segmentation(PCG,eng,nsamp,target_fs):
    assigned_states = eng.runSpringerSegmentationAlgorithmpython(PCG,matlab.double([target_fs]))
    idx_states,last_idx=eng.get_states_python(assigned_states,nargout=2)
    
    ncc=len(idx_states)
    idx_states=np.hstack(np.asarray(idx_states))
    idx_states=np.reshape(idx_states,(ncc,4))
    idx_states=idx_states-1 ## -1 for python indexing compatibility 
    last_idx=last_idx-1
    PCG = np.hstack(np.asarray(PCG))
    PCG = PCG/np.max(PCG)
    PCG = cast_to_floatx(PCG)
    x = np.zeros([ncc,nsamp],dtype=np.double)

    for row in range(ncc):
        if row == ncc-1:
            tmp = PCG[int(idx_states[row,0]):int(last_idx-1)] ## 2 to compensate for python indexing
        else:
            tmp=PCG[int(idx_states[row,0]):int(idx_states[row+1,0]-1)]
        N = nsamp-tmp.shape[0]
        x[row,:] = np.concatenate((tmp,np.zeros(N)))
    x = np.reshape(x,(x.shape[0],x.shape[1],1))
    return x


# In[29]:

@app.route("/predict",methods=["POST","GET"])
def predict():
    data={"success":False}
    input_request = flask.request.data
    if not input_request:
        return flask.jsonify(data)
    else:
        print(len(input_request))
        print(type(input_request))
        parsed = np.fromstring(input_request, np.int16)
        PCG = parsed.astype(np.float32) ## typecast for keras
        write('test.wav_',4000,PCG)
        PCG = matlab.double([np.ndarray.tolist(PCG)]) ## Typecast for matlab
        PCG = preprocessing(PCG=PCG,eng=eng,in_fs=in_fs,target_fs=target_fs)
        x = segmentation(PCG=PCG,eng=eng,nsamp=nsamp,target_fs=target_fs)
        y_pred=model.predict(x)
        print(y_pred)
        print(np.mean(y_pred))
        data["confidence"]=str(np.mean(y_pred))
        if np.mean(y_pred) > .5:
            print("Abnormal")
            data["success"] = True
            data["result"] = "Abnormal"
        else:
            print("Normal")
            data["success"] = True
            data["result"] = "Normal"
        # write('test.wav',4000,parsed)
        # plt.plot(np.fromstring(input_request,float))
        # dt=np.dtype(input_request)
        # print(dt.itemsize,dt.name)
        return flask.jsonify(data)


# @app.route("/predict",methods=["POST"])
# def predict():
#     data = {"success": False}
#     if flask.request.method=="POST":
#         input_request = flask.request.data
#         return flask.jsonify(data)
#     else:
#         return flask.jsonify({"error":"Couldn't understand request"})



# In[30]:


if __name__=='__main__':
    # app.run(host='127.0.1.2',debug=True,port=5000)
    eng = matlab_init()
    model = heartnet(load_path)
    app.run(host='0.0.0.0',port=5000) ## debug makes the process
                                    ## DONT debug and classfy
