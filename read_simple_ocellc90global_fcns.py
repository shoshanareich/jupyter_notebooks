import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from matplotlib import colors

import glob
plt.rcParams['figure.figsize'] = (12, 6)



#now that we understand how to expand the dim, we create a function.
#although, in this function, i'm also testing syntax for if statement
#so i decide to keep 2d and 3d separate, but ideally we should just
#check if nz==1, then we read, and then add a 3rd dim to the output fld.

#create a function
def read_llc90glob(filename,nx,ny,nz):
    with open(filename, 'rb') as f:
        data = np.fromfile(f, dtype=np.dtype('>f'))
        if nz > 1:
            fld = np.reshape(data,[nz, ny, nx])   #elif:
        else:
            fld = np.reshape(data,[ny, nx])
    return fld





#create a function
def plot_llc1245_faces(fld,nx,climit):
    clevels = np.arange(climit[0], climit[1], 1)
    fig,axs=plt.subplots(2,2)
    pcm=axs[0,0].contourf(fld[0,0:3*nx,0:nx],levels=clevels, cmap='viridis') #,vmin=-0.3,vmax=0.3,cmap='seismic')
    fig.colorbar(pcm,ax=axs[0,0],location='right')
    axs[0,0].title.set_text('fld face1')
    pcm=axs[0,1].contourf(fld[0,3*nx:6*nx,0:nx],levels=clevels,cmap='viridis')
    fig.colorbar(pcm,ax=axs[0,1],location='right')
    axs[0,1].title.set_text('fld face2')
    pcm=axs[1,0].contourf(np.reshape(fld[0,7*nx:10*nx,0:nx],[nx,3*nx]),levels=clevels,cmap='viridis')
    fig.colorbar(pcm,ax=axs[1,0],location='right')
    axs[1,0].title.set_text('fld face4')
    pcm=axs[1,1].contourf(np.reshape(fld[0,10*nx:13*nx,0:nx],[nx,3*nx]),levels=clevels,cmap='viridis')
    fig.colorbar(pcm,ax=axs[1,1],location='right')
    axs[1,1].title.set_text('fld face5')






#create a function
def patchface3D_test(fldin,nx,nz):
    print(fldin.shape)
    print(nz)
    #add a new dimension in case it's only 2d field:
    if nz == 1:
        fldin=fldin[np.newaxis, :, :]
    print(fldin.shape)
    
def patchface3D(fldin,nx,nz):
    
    print(nz)
    #add a new dimension in case it's only 2d field:
    if nz == 1:
        fldin=fldin[np.newaxis, :, :]
    
    #defining a big face:
    a=np.zeros((nz,4*nx,4*nx))       #(50,270,360)

    #face1
    tmp=fldin[:,0:3*nx,0:nx]        #(50,270,90)
    a[:,0:3*nx,0:nx]=tmp

    #face2
    tmp=fldin[:,(3*nx):(6*nx),0:nx] #(50, 270,90)
    a[:,0:3*nx,nx:2*nx]=tmp
    
    #face3
    tmp=fldin[:,(6*nx):(7*nx),0:nx] #(50, 90, 90)
    tmp=np.transpose(tmp, (1,2,0))  #(90, 90, 50)
    ##syntax to rotate ccw:
    tmp1=list(zip(*tmp[::-1]))
    tmp1=np.asarray(tmp1)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50, 90, 90)
    a[:,3*nx:4*nx,0:nx]=tmp1

    #face4
    tmp=np.reshape(fldin[:,7*nx:10*nx,0:nx],[nz,nx,3*nx]) #(50,90,270)
    tmp=np.transpose(tmp, (1,2,0))
    print(tmp.shape)                                      #(90,270,50)
    #syntax to rotate cw:
    tmp1=list(zip(*tmp))[::-1]      #type is <class 'list'>
    tmp1=np.asarray(tmp1)           #type <class 'numpy.ndarray'>, shape (270,90,50)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50,270,90)
    a[:,0:3*nx,2*nx:3*nx]=tmp1

    #face5
    tmp=np.reshape(fldin[:,10*nx:13*nx,0:nx],[nz,nx,3*nx]) #(50,90,270)
    tmp=np.transpose(tmp, (1,2,0))                         #(90,270,50)
    tmp1=list(zip(*tmp))[::-1]      #type is <class 'zip'> --> <class 'list'>
    tmp1=np.asarray(tmp1)           #type <class 'numpy.ndarray'>, shape (270,90,50)
    tmp1=np.transpose(tmp1,[2,0,1]) #(50,270,90)
    a[:,0:3*nx,3*nx:4*nx]=tmp1
    
    return a




#create a function
def patchface3D_vector(u_in,v_in,nx,nz,signswitchflag):
    
    #print(nz)
    #add a new dimension in case it's only 2d field:
    if nz == 1:
        u_in=u_in[np.newaxis, :, :]
        v_in=v_in[np.newaxis, :, :]
    
    #print(nz),print(u_in.shape),print(v_in.shape)
    
    fac = -1.0
    if signswitchflag == 0:
       fac = 1.0
        
    #print(fac)
    
    #=======================================
    #make 2 big arrays:
    u_out=np.zeros((nz,4*nx,4*nx+1))       #(nz,360,361)
    v_out=np.zeros((nz,4*nx+1,4*nx))       #(nz,361,360)
    print(u_out.shape),print(v_out.shape)

    #=======================================
    #face1: just fill in
    u_out[:,0:3*nx,0:nx]=u_in[:,0:3*nx,0:nx]        #(nz,270,90)
    v_out[:,0:3*nx,0:nx]=v_in[:,0:3*nx,0:nx]

    #=======================================
    #face2: just fill in
    u_out[:,0:3*nx,nx:2*nx]=u_in[:,3*nx:6*nx,0:nx]
    v_out[:,0:3*nx,nx:2*nx]=v_in[:,3*nx:6*nx,0:nx]
    
    #=======================================
    #face3
    #v becomes -u, with a shift,
    tmpv=v_in[:,(6*nx):(7*nx),0:nx]             #(nz, 90, 90)
    tmpv=np.transpose(tmpv, (1,2,0))            #(90,90,nz)
    ##syntax to rotate ccw:
    tmp1u=list(zip(*tmpv[::-1]))
    tmp1u=np.asarray(tmp1u)
    tmp1u=np.transpose(tmp1u,[2,0,1])            #(nz,90,90)    
    u_out[:,3*nx:4*nx,1:nx+1]=tmp1u                 
    #note shift in x above, so first cell is missing, which is the first column of face 5; skip for now
    
    #u becomes v, need to reverse order
    tmpu=u_in[:,(6*nx):(7*nx),0:nx]             #(nz, 90, 90)
    tmpu=np.transpose(tmpu, (1,2,0))           #(90,90,nz)
    ##syntax to rotate ccw:
    tmp1v=list(zip(*tmpu[::-1]))
    tmp1v=np.asarray(tmp1v)
    tmp1v=np.transpose(tmp1v,[2,0,1])            #(nz,90,90)
    v_out[:,3*nx:4*nx,0:nx]=fac*tmp1v
    
    #======================================    
    #face4: 
    #v becomes u, need to reverse order
    tmpv=np.reshape(v_in[:,7*nx:10*nx,0:nx],[nz,nx,3*nx]) #(nz,90,270)
    tmpv=np.transpose(tmpv, (1,2,0))                      #(90,270,nz)
    #syntax to rotate cw:
    tmp1u=list(zip(*tmpv))[::-1]                        #type is <class 'list'>
    tmp1u=np.asarray(tmp1u)                             #type <class 'numpy.ndarray'>, shape (270,90,nz)
    tmp1u=np.transpose(tmp1u,[2,0,1])                   #(nz,270,90)
    u_out[:,0:3*nx,2*nx:3*nx]=tmp1u                     #np.flip(tmp1u,axis=0)  #[3*nx:None:-1,:]  #note NOT flipping

    #u becomes -v, with a shift, 
    tmpu=np.reshape(u_in[:,7*nx:10*nx,0:nx],[nz,nx,3*nx])  #(nz,90,270)
    tmpu=np.transpose(tmpu, (1,2,0))                    #(90,270,nz)
    tmp1v=list(zip(*tmpu))[::-1]
    tmp1v=np.asarray(tmp1v)
    tmp1v=np.transpose(tmp1v,[2,0,1])                   #(nz,270,90)
    v_out[:,1:3*nx+1,2*nx:3*nx]=fac*tmp1v               #should not be flipping, [3*nx:None:-1,:], #note minus one
    
    #======================================
    #face5: v becomes u
    tmpv=np.reshape(v_in[:,10*nx:13*nx,0:nx],[nz,nx,3*nx]) #(nz,90,270)
    tmpv=np.transpose(tmpv, (1,2,0))                    #(90,270,nz)
    #syntax to rotate cw:
    tmp1u=list(zip(*tmpv))[::-1]                        #type is <class 'list'>
    tmp1u=np.asarray(tmp1u)                             #type <class 'numpy.ndarray'>, shape (270,90,nz)
    tmp1u=np.transpose(tmp1u,[2,0,1])                   #(nz,270,90)
    u_out[:,0:3*nx,3*nx:4*nx]=tmp1u                     #np.flip(tmp1u,axis=0)   #[3*nx:None:-1,:]  #note NOT flipping

    #face5: u becomes -v, with a shift, 
    tmpu=np.reshape(u_in[:,10*nx:13*nx,0:nx],[nz,nx,3*nx]) #(nz,90,270)
    tmpu=np.transpose(tmpu, (1,2,0))                    #(90,270,nz)
    #syntax to rotate cw:
    tmp1v=list(zip(*tmpu))[::-1]                        #
    tmp1v=np.asarray(tmp1v)                             #(270,90,nz)
    tmp1v=np.transpose(tmp1v,[2,0,1])                   #(nz,270,90)
    v_out[:,1:3*nx+1,3*nx:4*nx]=fac*tmp1v               #should not be flipping [3*nx:None:-1,:], #note minus one
    
    return u_out, v_out




#recordLen = chunk_size     # 
#recordSize = recordLen * data_type # size of a record in bytes
#memArray = np.zeros(recordLen, dtype=np.dtype('>f' + str(data_type))) # a buffer for 1 record
#del memArray, tmp
def read_chunk(filename,chunk_size,recordNo,data_type):
    with open(filename, 'rb') as file:
        recordSize=chunk_size * data_type
        # Reading a record recordNo from file into the memArray
        file.seek(recordSize * recordNo)
        bytes = file.read(recordSize)
        memArray = np.frombuffer(bytes, dtype=np.dtype('>f' + str(data_type))).copy()
    return memArray





def read_chunk2(filename,nx,tile_y,recordNo,data_type):
    with open(filename, 'rb') as file:
        recordSize=tile_y*nx * data_type
        # Reading a record recordNo from file into the memArray
        file.seek(recordSize * recordNo)
        bytes = file.read(recordSize)
        memArray = np.frombuffer(bytes, dtype=np.dtype('>f' + str(data_type))).copy()
    return memArray



