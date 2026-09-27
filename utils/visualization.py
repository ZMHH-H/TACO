import numpy as np
import torch

def show_video_frame(video_decode,window_name='video_frame',crop_box=None,img_ori_size=(320,240)):  
    import cv2 
    # 复制，送到cpu，去掉第一维度，Cxframe_numberxHxW->frame_numberxHxWxC,反归一化
    img0 = video_decode.clone().detach().cpu().squeeze().permute(1,2,3,0).mul(255)
    # img_interval = numpy.ones((224,32,3))
    # 图像之间的白色间隔
    img_interval = np.fromfunction(lambda i,j,k:(i+j+k)*0+255,(img0.shape[1],32,3),dtype=np.uint8)
    # 总图像
    final_img = np.fromfunction(lambda i,j,k:(i+j+k)*0+255,(img0.shape[1],32,3),dtype=np.uint8)
    for i in range(img0.shape[0]):
        img_np = img0[i].type(torch.uint8).numpy()
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        if crop_box is not None:
            img_np = cv2.rectangle(img_np,(int(crop_box[0]*img_ori_size[0]),int(crop_box[1]*img_ori_size[1])),(int(crop_box[2]*img_ori_size[0]),int(crop_box[3]*img_ori_size[1])),(255,0,0),2)
        final_img = np.hstack((final_img,img_np,img_interval))
    
    print('final_img',final_img.shape)
    final_img = cv2.resize(final_img,(0,0),fx=1,fy=1)
    cv2.imshow(window_name,final_img)
    cv2.waitKey(500000)