import numpy as np

def read_float64(fileIn):
    with open(fileIn, 'rb') as f:
        data = np.fromfile(f, dtype=np.dtype('>f8'))
        print(np.shape(data))
    return data

def read_float32(fileIn):
    with open(fileIn, 'rb') as f:
        data = np.fromfile(f, dtype=np.dtype('>f'))
        print(np.shape(data))
    return data

def write_float32(fout,fld):
    with open(fout, 'wb') as f:
        np.array(fld, dtype=">f").tofile(f)

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

niter = '0000815256'

#dirrun = '/nobackup/dcarrol2/v05_latest/darwin3/run/'
dirrun = '/nobackup/sreich/llc270_c68w_runs/run_pk0000815256_300s_nouv/'
pickup = read_float64(dirrun + 'pickup.' + niter + '.data')


nx = 270
ny = 3510
nz = 50

#theta, salt, uvel, vvel 3d
#etan 2d

uvel = pickup[:nx*ny*nz]

vvel = pickup[nx*ny*nz:2*nx*ny*nz]

theta = pickup[2*nx*ny*nz:3*nx*ny*nz]

salt = pickup[3*nx*ny*nz:4*nx*ny*nz]

#g = pickup[4*nx*ny*nz:8*nx*ny*nz] # g's are 3d fields

etan = pickup[8*nx*ny*nz:8*nx*ny*nz + nx*ny]


dirtemp = '/nobackup/sreich/llc270_c68w_runs/run_template/'

write_float32(dirtemp + 'U.' + niter + '.data', uvel)
write_float32(dirtemp + 'V.' + niter + '.data', vvel)
write_float32(dirtemp + 'Theta.' + niter + '.data', theta)
write_float32(dirtemp + 'Salt.' + niter + '.data', salt)
write_float32(dirtemp + 'Eta.' + niter + '.data', etan)








si = read_float64(dirrun + 'pickup_seaice.' + niter + '.data')
# all are 2d fields

# area = si[:nx*ny]
# snow = si[nx*ny:2*nx*ny]
# salt = si[2*nx*ny:3*nx*ny]
# heff = si[3*nx*ny:4*nx*ny]
# uice = si[4*nx*ny:5*nx*ny]
# vice = si[5*nx*ny:6*nx*ny]

# tices = si[:nx*ny]
area = si[nx*ny:2*nx*ny]
heff = si[2*nx*ny:3*nx*ny]
snow = si[3*nx*ny:4*nx*ny]
salt = si[4*nx*ny:5*nx*ny]
uice = si[5*nx*ny:6*nx*ny]
vice = si[6*nx*ny:7*nx*ny]

write_float32(dirtemp + 'SIarea.' + niter + '.data', area)
write_float32(dirtemp + 'SIhsnow.' + niter + '.data', snow)
write_float32(dirtemp + 'SIhsalt.' + niter + '.data', salt)
write_float32(dirtemp + 'SIheff.' + niter + '.data', heff)
write_float32(dirtemp + 'SIuice.' + niter + '.data', uice)
write_float32(dirtemp + 'SIvice.' + niter + '.data', vice)


