import numpy as np
def add_pad(X, pad):
    """
    add padding to X
    
    Argument:
    X --Shape=(m_data,n_H,n_W,n_d) tương ứng là số ảnh , chiều cao, chiều rồng và độ sâu
    Returns:
    X_pad -- image : shape (m, n_H + 2*pad, n_W + 2*pad, n_C)
    """
    X_pad =np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant',constant_values = 0)
    return X_pad.astype(float)


def move_conv2d(conv2d_,w,bias=0):
    s=np.multiply(conv2d_,w)
    Z=np.sum(s,axis=None)+bias
    return Z

def cal_conv2d(A_pre,W,bias,**kwargs):
    '''
        A_pre la input 4 tensor
        w la para input 4 tensor
        bias la para input 4 tensor shape(1,1,1,w_cur.shape[3])
        args la dictionary {'strides':int,'pad':int}
    '''
    (m,n_H,n_W,n_D)=A_pre.shape
    (f, f, n_D_pre,n_D_cur)=W.shape
    
    # layer pre phai conected voi layer hien tai thong qua 1 dim giong voi mang neuralnet
    if n_D != n_D_pre:
        raise Exception('erro layer in Neural Net shape prelayer : {}{}{}{} and shape cur layer {}{}{}{}'.format(m,n,n_W,n_D,f,f,n_D_pre,n_D_cur))
    
    strides=kwargs['strides']
    pad=kwargs['pad']
    
    new_H=np.int((n_H-f+pad*2)/(strides))+1
    new_W=np.int((n_W-f+pad*2)/(strides))+1
    
    Z=np.zeros((m,new_H,new_W,n_D_cur))
    
    A_with_pading=add_pad(A_pre,pad)
    # print(A_pre.shape,A_with_pading.shape)
    for i in range(m):
         # duyet qua m image
        image_here=A_with_pading[i,:,:,:]
        for h in range(new_H):
            for w in range(new_W):
                for c in range(n_D_cur):
                    x_s=h*strides
                    x_e=h*strides+f
                    y_s=w*strides
                    y_e=w*strides+f
                    #print(x_s,x_e,y_s,y_e,image_here.shape,A_with_pading.shape)
                    Z[i,h,w,c]=move_conv2d(image_here[x_s:x_e,y_s:y_e,:],W[:,:,:,c],bias[:,:,:,c])
                    
    return Z
    
def cal_pool(A_prev,  mode = "max",**kwargs,):
    (m,n_Hp,n_Wp,n_Dp)=A_prev.shape
    
    f=kwargs['kernel_size']
    strides=kwargs['strides']
    n_H = np.int(1 + (n_Hp - f) / strides) # giong voi conv voi padding=0 strides =2 
    # Li do 1 so mang conv2 thay maxpooling = conv2d voi activation=liner va strides=2 padding =0
    n_W = np.int(1 + (n_Wp - f) / strides)
    n_C = n_Dp # deep k thay doi
    
    A=np.zeros((m,n_H,n_W,n_C))
    for i in range(m):
        image_here=A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    x_s=h*strides
                    x_e=h*strides+f
                    y_s=w*strides
                    y_e=w*strides+f
                    if mode=='max': #maxpooling
                        A[i,h,w,c]=np.max(image_here[x_s:x_e,y_s:y_e,c])
                    elif mode=='average':
                        A[i,h,w,c]=np.mean(image_here[x_s:x_e,y_s:y_e,c])
    return A


def conv2d_backward(dZ, value_back,**kwagrs):

    A_pre,W,bias=value_back
    
    (m,n_H,n_W,n_D)=A_pre.shape
    (f, f, n_D_pre,n_D_cur)=W.shape
    
    # layer pre phai conected voi layer hien tai thong qua 1 dim giong voi mang neuralnet
    if n_D != n_D_pre:
        raise Exception('erro layer in Neural Net shape prelayer : {}{}{}{} and shape cur layer {}{}{}{}'.format(m,n,n_W,n_D,f,f,n_D_pre,n_D_cur))
    
    strides=kwagrs['strides']
    pad=kwagrs['pad']
    
    new_H=np.int((n_H-f+pad*2)/(strides))+1
    new_W=np.int((n_W-f+pad*2)/(strides))+1
    
    
    A_with_pading=add_pad(A_pre,pad)
    d_w=np.zeros_like(W)
    d_b=np.zeros_like(bias)
    d_a=np.zeros_like(A_pre) # dao ham cua  1 ma tran la  1 ma tran cung chieu voi no
    d_a_pad=np.zeros_like(A_with_pading)
    # print(A_pre.shape,A_with_pading.shape)
    for i in range(m):
         # duyet qua m image
        image_here=A_with_pading[i,:,:,:]
        d_image_here=d_a_pad[i,:,:,:]
        for h in range(new_H):
            for w in range(new_W):
                for c in range(n_D_cur):
                    x_s=h*strides
                    x_e=h*strides+f
                    y_s=w*strides
                    y_e=w*strides+f
                    d_image_here[x_s:x_e,y_s:y_e,:]+=W[:,:,:,c]*dZ[i,h,w,c]
                    d_w[:,:,:,c]+=image_here[x_s:x_e,y_s:y_e,:]*dZ[i,h,w,c]
                    d_b[:,:,:,c]+=dZ[i,h,w,c]
        d_a[i,:,:,:]=d_image_here[pad:-pad,pad:-pad,:]
    return (d_a,d_w,d_b)
                    


def pool_backward(dA,value_back,mode='max',**kwagrs):
    (A_prev) = value_back
    (m,n_Hp,n_Wp,n_Dp)=A_prev.shape
    f=kwagrs['kernel_size']
    strides=kwagrs['strides']
    n_H = np.int(1 + (n_Hp - f) / strides)
    n_W = np.int(1 + (n_Wp - f) / strides)
    n_C = n_Dp # deep k thay doi
    
    #A=np.zeros((m,n_H,n_W,n_C))
    d_A=np.zeros_like(A_prev)
    for i in range(m):
        image_here=A_prev[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    x_s=h*strides
                    x_e=h*strides+f
                    y_s=w*strides
                    y_e=w*strides+f
                    if mode=='max': #maxpooling
                        conv2d_here=image_here[x_s:x_e,y_s:y_e,c]
                        ismax=(conv2d_here==np.max(conv2d_here))
                        d_A[i,x_s:x_e,y_s:y_e,c]+=ismax*dA[i,h,w,c]
                    elif mode=='average':
                        raise Exception('k phai max_pooling')
                        return None
    return d_A
    