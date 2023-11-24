from typing import Union
from pydantic import BaseModel
from app.tracker import BallTracker
import os
import random
import tensorflow_hub as hub
from sklearn.preprocessing import minmax_scale
from skimage.measure import LineModelND
from skimage.measure import ransac as ran
from circle_fit import taubinSVD
from fastapi import FastAPI
import numpy as np
import cv2
import math
from circle_fit import taubinSVD
import tensorflow as tf

detector = hub.load("https://tfhub.dev/tensorflow/efficientdet/lite4/detection/1")



def pointInRect(rect, points) :
    inside_pts = []
    x1, y1, x2,y2 = rect[0][0],rect[0][1],rect[1][0],rect[1][1]
    for p in points:
        x,y = p
        if (x1 < x and x < x2):
            if (y1 < y and y < y2):
                inside_pts.append(p)
    return inside_pts 

def createLine(pts,min_samples_,res):
    model = LineModelND()
    model.estimate(pts)
    model_robust, inliers = ran(pts, LineModelND, min_samples=min_samples_,residual_threshold=res, max_trials=100)
    inliers = np.where(inliers==True)[0]
    
    outliers = np.array([pts[i] for i in range(len(pts)) if i not in inliers])
    x_array = np.array([pts[inliers[0]][0],pts[inliers[-1]][0]])
    y_array = np.array([pts[inliers[0]][1],pts[inliers[-1]][1]])
    slope = (y_array[0] - y_array[1])/(x_array[0]-x_array[1])
    return x_array,y_array,outliers,slope



def getLinesinGrid(img,grid_shape, fg_centers,roi_pts,batsman_roi,orientation):
    
    h, w, _ = img.shape
    rows, cols = grid_shape
    dy, dx = h / rows, w / cols
    color = (255,255,255)
    thickness = 1

    for x in np.linspace(start=dx, stop=w, num=cols):
        x = int(round(x))
        cv2.line(img, (x,0), (x, h), color=color, thickness=thickness)

    for y in np.linspace(start=dy, stop=h, num=rows):
        y = int(round(y))
        cv2.line(img, (0, y), (w, y), color=color, thickness=thickness)
       
    
    prev_pt = (0,0)
    start_prev_pt = (0,0)

    slope_pts = []
    for id_x,x in enumerate(np.linspace(start=dx, stop=w, num=cols)):
        count = 0
        col_pts = []
        for id_y,y in enumerate(np.linspace(start=dy, stop=h, num=rows)):
        
            rect_splice = [prev_pt,(x,y)]
            bl_splice = pointInRect(rect_splice,fg_centers)
            color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
            for pt in bl_splice:
                cv2.circle(img,(int(pt[0]),int(pt[1])) , 4 , (255,0,0) ,-1)
            '''
            try bakchodi here
            '''
            new_row = np.array(bl_splice)
            samples = len(new_row)
            while samples > 2:
                x_array,y_array,outliers,slope = createLine(new_row,2,2)
                y_ = [y_array[0],y_array[1]]
                x_ = [x_array[0],x_array[1]]
                if orientation == 0:
                    if slope > 0.2:
                        p1 = np.asarray([x_[0],y_[0]])
                        p2 = np.asarray([x_[1],y_[1]])
                        p3 = np.asarray(new_row)
                        d=np.absolute(np.cross(p1-p2,p3-p1)/np.linalg.norm(p1-p2))
                        d_ = np.where((d < 3 ) & (d >= 0))[0]
                        line_pts = [new_row[i] for i in d_]
                        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                        slope_pts_=[]
                        for pt in line_pts:
                            cv2.circle(img,(int(pt[0]),int(pt[1])) , 4 , color ,-1)
                            cv2.circle(img,(int(pt[0]),int(pt[1])) , 2 , (255,255,255) ,-1)
                            slope_pts_.append(pt)
                        slope_pts.append(slope_pts_)
                else:
                    if slope < -0.2:
                        p1 = np.asarray([x_[0],y_[0]])
                        p2 = np.asarray([x_[1],y_[1]])
                        p3 = np.asarray(new_row)
                        d=np.absolute(np.cross(p1-p2,p3-p1)/np.linalg.norm(p1-p2))
                        d_ = np.where((d < 3 ) & (d >= 0))[0]
                        line_pts = [new_row[i] for i in d_]
                        color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
                        slope_pts_=[]
                        for pt in line_pts:
                            cv2.circle(img,(int(pt[0]),int(pt[1])) , 4 , color ,-1)
                            cv2.circle(img,(int(pt[0]),int(pt[1])) , 2 , (255,255,255) ,-1)
                            slope_pts_.append(pt)
                        slope_pts.append(slope_pts_)
                        
                samples = len(outliers)
                new_row = outliers

                col_pts.append(np.array(bl_splice))

            count = count + 1
            if count  == rows:
                prev_pt = (start_prev_pt[0],0)
                count = 0
            elif count == 1:
                prev_pt = (prev_pt[0],y)
                start_prev_pt = (x,y)
            else:
                prev_pt = (prev_pt[0],y)
                
    cv2.rectangle(img,batsman_roi[0],batsman_roi[1],(0,255,0),1)
    return img,slope_pts



def perspectiveWrap(base,target,bg_points,hg_points):
    ###pitch points should be in tl tr br bl
    h,w,c = base.shape
    h_t,w_t,c_t = target.shape
    target_pts = np.float32([(0,0),(w_t,0),(w_t,h_t),(0,h_t)])
    target_pts = np.float32(hg_points)
    base_pts = np.float32(bg_points)
    M = cv2.getPerspectiveTransform(target_pts,base_pts)
    dst = cv2.warpPerspective(target,M,(w,h))

    mask = np.zeros(base.shape, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(base_pts), (255, 255, 255))
    mask = cv2.bitwise_not(mask)
    masked_image = cv2.bitwise_and(base, mask)
    output = cv2.bitwise_or(dst, masked_image)
    
    return output,M

def drawTrajectory(input_vid,output_vid,circle,pitch_coords,hg_pitch,hg_pitch_coords,bat_pos,rect,cameraPos):
    
    xc,yc,r = circle
    bat_x,bat_y = bat_pos

    if cameraPos ==0:
        bat_x=int(xc + np.sqrt(r*r - (bat_y-yc)*(bat_y-yc)))
    else:
        bat_x=int(xc - np.sqrt(r*r - (bat_y-yc)*(bat_y-yc)))

    alpha = 0.6
    beta = 1-alpha
    cap = cv2.VideoCapture(input_vid)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    numberOfFrames = int(cap.get(7))


    for y in np.linspace(0,200,5):
        try:

            if cameraPos ==0:
                init = (int(xc + np.sqrt(r*r - (y-yc)*(y-yc))),y)
            else:
                init = (int(xc - np.sqrt(r*r - (y-yc)*(y-yc))),y)
            
            if init[0]!= None:
                break
        except:
            continue

    start_angle = math.degrees(math.atan2(init[1] - yc, init[0] - xc))
    

    all_imgs = []
    fg_writer = cv2.VideoWriter(output_vid,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
    for id__y, y_ in enumerate(np.linspace(init[1],bat_y,numberOfFrames)):
        ret, frame = cap.read()
        if id__y == 0:
            '''
            perspective transform the bounce point
            '''
            output,M = perspectiveWrap(frame,hg_pitch,pitch_coords,hg_pitch_coords)
            bat_x , bat_y = bat_x + rect[0][0],bat_y + rect[0][1]
            A = np.array([[[bat_x, bat_y]]], dtype=np.float32)
            
            pt2 = cv2.perspectiveTransform(A,np.linalg.inv(M))[0][0]
            cv2.circle(hg_pitch,(int(pt2[0]),int(pt2[1])),5,(0,0,0),2)
            cv2.circle(hg_pitch,(int(pt2[0]),int(pt2[1])),3,(255,255,255),-1)
            print('perspective are :: '  + str(pt2))
            # y_perspective = frame_height - int(hg_pitch.shape[0]/2)
            # x_perspective = frame_width  - int(hg_pitch.shape[1]/2)
            fg_writer.write(output)

        
        h__,w__ = rect[1][1] -rect[0][1],rect[1][0]-rect[0][0]
        img_ = np.zeros((h__, w__, 3), np.uint8)

        if cameraPos ==0:
            final = (int(xc + np.sqrt(r*r - (y_-yc)*(y_-yc))),int(y_))
        else:
            final = (int(xc - np.sqrt(r*r - (y_-yc)*(y_-yc))),int(y_))
        
        start__ = start_angle
        for idx,inter_y in enumerate(np.linspace(init[1],y_,4)):
            color = (255,255,255)
            thick = 1
            if idx == 1:
                color = (0,0,255)
                thick = 6
            if idx == 2:
                color = (0,0,255)
                thick = 5
            if idx == 3:
                color = (0,0,255)
                thick = 4


            if cameraPos ==0:
                final = (int(xc + np.sqrt(r*r - (inter_y-yc)*(inter_y-yc)) ),int(inter_y))
            else:
                final = (int(xc - np.sqrt(r*r - (inter_y-yc)*(inter_y-yc)) ),int(inter_y))
                
            end_angle = math.degrees(math.atan2(final[1] - yc,final[0] - xc))
            cv2.ellipse(img_, (int(xc),int(yc)), (int(r),int(r)), 0,int(start__), int(end_angle), color, thick)
            start__ = end_angle

        
        cv2.circle(img_,final,10,(255,255,255),-1)
        all_imgs.append(img_)
        frame_crop  = frame[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]] 
        # print(frame_crop.shape)
        # print(img_.shape)
        dst = cv2.addWeighted(frame_crop, alpha, img_, beta, 0.0)
        frame[rect[0][1]:rect[1][1],rect[0][0]:rect[1][0]]  = dst
        frame[int(frame_height-hg_pitch.shape[0]):frame_height,int(frame_width-hg_pitch.shape[1]):frame_width] = hg_pitch
        fg_writer.write(frame)


    fg_writer.release()
    cap.release()
    

app = FastAPI()

class Item(BaseModel):
    path: str
    roi: list
    pitch: list
    minArea: list
    maxArea:float
    refinedMinArea:float
    bgs:int
    cameraPos:int
    rootDir: str

class Traj(BaseModel):
    rootDir:str
    path: str
    roi: list
    pitch: list
    pts: list
    cameraPos: int

    
@app.post("/cropROI")
def crop_roi(item : Item):
 
    abs_vid_path = '/Videos/'+item.path
    output_path = '/Videos/bgs/'+item.path.split('.')[0] + '.avi'
    tracker = BallTracker(abs_vid_path,cam_orientation='left',batsman_orientation='rhb',roi=item.roi,hg_pitch='',hg_pitch_coords='',pitch_coords=item.pitch,players=2)
    tracker.runFrames([0,0,0])
    tracker.backgroundSubtraction([0,0,0],1,7,output_path)
    return {"path": output_path}

@app.post("/getContours")
def get_contours(item : Item):
    abs_vid_path = '/Videos/'+item.path
    output_path = '/Videos/contours/'+item.path.split('.')[0] + '.avi'
    tracker = BallTracker(abs_vid_path,cam_orientation='left',batsman_orientation='rhb',roi=item.roi,hg_pitch='',hg_pitch_coords='',pitch_coords=item.pitch,players=2)
    tracker.runFrames([0,0,0])
    tracker.backgroundSubtraction([0,0,0],0,7,output_path)
    tracker.getContours(1000,[50,300],1.5,[0,0,0],1,output_path)
    return {"path": output_path}


@app.post("/finalContours")
def createTrajectory(item : Traj):
    abs_vid_path = item.path
    localFile = abs_vid_path.split('/')[-1]
    justFile = localFile.split('.')[0]
    debug_path = item.rootDir + '/indiContours/'+justFile + '.avi'

    xmin,ymin,xmax,ymax = item.roi[0][0],item.roi[0][1],item.roi[1][0],item.roi[1][1]
    a,b, r, sigma = taubinSVD(item.pts)
    cap = cv2.VideoCapture(abs_vid_path)
    ret,frame = cap.read()
    frame = frame[ymin:ymax,xmin: xmax]
    tensor = tf.convert_to_tensor(frame)
    tensor = tf.expand_dims(tensor, axis=0)
    boxes, scores, classes, num_detections = detector(tensor)
    c  = np.where(classes[0] == 1)[0]
    person_boxes = []
    updated = False
    for person in c:
        if scores[0][person] > 0.3:
            ymax,xmax,ymin,xmin = boxes[0][person]
            batsman_roi = [(xmin,ymin),(xmax,ymax)]

    hg_pitch_coords = [(20,112),(188,112),(188,475),(20,475)]
    hg_pitch_coords = [(10,56),(94,56),(95,238),(10,238)]
    hg_pitch = cv2.imread('/Users/muck27/Downloads/tensorflow/app/hg-pitch.png')
    orig_h,orig_w,c = hg_pitch.shape
    print(orig_h,orig_w)
    scaled_h,scaled_w = int(orig_h/2),int(orig_w/2)
    print(scaled_h,scaled_w)
    hg_pitch = cv2.resize(hg_pitch,(scaled_w,scaled_h) ,interpolation = cv2.INTER_LINEAR)    

    bat_x,bat_y = batsman_roi[0][0],batsman_roi[0][1]
    
    drawTrajectory(item.path,debug_path,[a,b,r],item.pitch,hg_pitch,hg_pitch_coords,(bat_x,bat_y),item.roi,item.cameraPos)
    return debug_path

@app.post("/getDebugContours")
def get_debug_contours(item : Item):
    
    abs_vid_path = item.path
    localFile = abs_vid_path.split('/')[-1]
    justFile = localFile.split('.')[0]
    debug_path = item.rootDir + '/indiContours/'+justFile + '.avi'
    ctr_path = item.rootDir +'/contours/'+justFile + '.avi'
    bgs_path = item.rootDir +'/bgs/'+justFile + '.avi'
    output_ctr_path = item.rootDir +'/indiContours/' + justFile + '.jpg'

    # print(abs_vid_path)
    # print(debug_path)
    # print(ctr_path)
    # print(bgs_path)
    # print(output_ctr_path)
    # pts = [(1,2),(2,3)]

    hg_pitch_coords = [(20,112),(188,112),(188,475),(20,475)]
    # hg_pitch_coords = [(10,56),(94,56),(95,238),(10,238)]
    hg_pitch = cv2.imread('/Users/muck27/Downloads/tensorflow/app/hg-pitch.png')
    # orig_h,orig_w,c = hg_pitch.shape
    # print(orig_h,orig_w)
    # scaled_h,scaled_w = int(orig_h/2),int(orig_w/2)
    # print(scaled_h,scaled_w)
    # hg_pitch = cv2.resize(hg_pitch,(scaled_w,scaled_h) ,interpolation = cv2.INTER_LINEAR)    


    minArea = item.minArea
    maxArea = item.maxArea
    refinedMinArea = item.refinedMinArea
    bgs = item.bgs
    roi = item.roi
    pitch_coords = item.pitch
    cameraPos = item.cameraPos


    tracker = BallTracker(abs_vid_path,cam_orientation='left',batsman_orientation='rhb',roi=roi,hg_pitch='',hg_pitch_coords='',pitch_coords=pitch_coords,players=2)
    tracker.runFrames([0,0,0])
    tracker.backgroundSubtraction([0,0,0],1,bgs,bgs_path,4,1)
    tracker.getContours(1000,[maxArea,maxArea],refinedMinArea,[0,0,0],1,ctr_path)
    tracker.debugContours(debug_path)
    start_frame = tracker.getAllDetections(detector,2)
    # tracker.viewContours([0,0],output_ctr_path)


    
    pts,batsman_roi,pts_with_radii = tracker.removeNoise(7,False,True)   
    h,w = roi[1][1] -roi[0][1],roi[1][0]-roi[0][0]
    roi_img = np.zeros((h,w,3),np.uint8)
    img,slope_pts =   getLinesinGrid(roi_img, (int(minArea[0]),int(minArea[1])),pts_with_radii,6,batsman_roi,cameraPos)

    # better_pts = []
    # for idx,slope in enumerate(slope_pts):
    #     if (len(slope)) > 2:
    #         a,b, r, sigma = taubinSVD(slope)
    #         if a!= math.isinf(a) and b!=math.isinf(b) and r !=math.isinf(r) and b > slope[0][1]:
    #             better_pts.append(slope)


    # if len(better_pts) >0:
    #     largest = np.asarray(max(enumerate(better_pts), key=(lambda x: len(x[1]))))
    #     others =  np.asarray([better_pts[i] for i in range(len(better_pts)) if i !=largest[0]])
    #     merge = largest[1]
    #     for i in range(len(others)):
    #         other = others[i]
    #         new = [pt for pt in merge]
    #         for pt in other:
    #             new.append(pt)
    #         a,b, r, sigma = taubinSVD(new)
    #         if sigma < 1.5:
    #             merge =new
        
    #     a,b, r, sigma = taubinSVD(merge)
    #     cv2.circle(img,(int(a),int(b)),int(r),(0,0,255),2)
    
    #     bat_x,bat_y = batsman_roi[0][0],batsman_roi[0][1]
    #     drawTrajectory(abs_vid_path,debug_path,[a,b,r],pitch_coords,hg_pitch,hg_pitch_coords,(bat_x,bat_y),roi,cameraPos)


    cv2.imwrite(output_ctr_path,img)

    return {"bgs": bgs_path,'image':output_ctr_path,'contour':ctr_path,'debug':debug_path,'pts':pts,'radius':pts_with_radii}

