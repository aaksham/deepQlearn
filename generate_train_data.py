import gym
import time
import numpy
import joblib
import imagehash
from cStringIO import StringIO
from collections import defaultdict
import PIL.Image
from deeprl_hw2 import policy

def prepro(I):
    """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
    I = I[35:195] # crop
    I = I[::2,::2,0] # downsample by factor of 2
    I[I == 144] = 0 # erase background (background type 1)
    I[I == 109] = 0 # erase background (background type 2)
    I[I != 0] = 1 # everything else (paddles, ball) just set to 1
    I=I.astype(numpy.float)
    return I

env=gym.make('SpaceInvaders-v0')
hashfunc=imagehash.average_hash
hashsize=80

state_hash_table=defaultdict(list)
experience=[]

start=env.reset()
runs=500
st=start
start_time=time.time()

for play in range(runs):
    st_processed=prepro(st)
    st_image=PIL.Image.fromarray(st_processed)
    current_state=hashfunc(st_image,hash_size=hashsize)
    state_hash_table[current_state].append(st_image)
    at=env.action_space.sample()
    next_tuple=env.step(at)
    st1=next_tuple[0]
    rt=next_tuple[1]
    isterminal=next_tuple[2]
    st1_processed=prepro(st1)
    st1_image=PIL.Image.fromarray(st1_processed)
    next_state=hashfunc(st1_image,hash_size=hashsize)
    state_hash_table[next_state].append(st1_image)
    et=(current_state,at,rt,next_state)
    experience.append(et)
    if isterminal: st=env.reset()

print ((time.time()-start_time)/60)

print len(state_hash_table.keys())

for key in state_hash_table.keys():
    if len(state_hash_table[key])!=1:
        if len(state_hash_table[key])==2:
            i=1
            for e in state_hash_table[key]:
                #print e.shape
                earray=numpy.array(e)
                final=PIL.Image.fromarray(earray.astype(numpy.uint8))
                final.save(str(i)+'.png')
                print type(earray)
                # f=StringIO()
                # e.save(f,'jpeg')
                i+=1
            break

joblib.dump(experience,'replay/replay.pkl')
joblib.dump(state_hash_table,'states/state_hash.pkl')
# f=h5py.File("replay.hdf5","w")
# f.create_dataset('data',data=experience)
# f.close()
#
# f=h5py.File("state_hash","w")
# f.create_dataset('data',data=state_hash_table)
# f.close()
#



