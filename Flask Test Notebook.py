
# coding: utf-8

# In[28]:


import matlab.engine
from heartnet_v1 import heartnet
import numpy as np
import flask
import io

app=flask.Flask(__name__)
model=None

load_path='/media/taufiq/Data/heart_sound/models/fold1_noFIR 2018-02-02 09:52:02.463256/weights.0148-0.8902.hdf5'


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
    eng.addpath('/media/taufiq/Data/heart_sound/Heart_Sound/codes/cristhian.potes-204/');
    return eng


# In[25]:


def segmentation(PCG,eng,nsamp):
    assigned_states = eng.runSpringerSegmentationAlgorithmpython(PCG,matlab.double([target_fs]))
    idx_states,last_idx=eng.get_states_python(assigned_states,nargout=2)
    
    ncc=len(idx_states)
    idx_states=np.hstack(np.asarray(idx_states))
    idx_states=np.reshape(idx_states,(ncc,4))
    idx_states=idx_states-1 ## -1 for python indexing compatibility 
    last_idx=last_idx-1
    PCG = np.hstack(np.asarray(PCG))
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


@app.route("/predict",methods=["POST"])
def predict():
    data = {"success": False}
    if flask.request.method=="POST":
        return flask.jsonify(data)



# In[30]:


if __name__=='__main__':
    app.run(debug=True,port=5000)

