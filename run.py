
import os
import sys

import numpy as np
import tensorflow as tf
import cv2
from numpy.linalg import inv

import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#################################################################################################################

def load_tracking_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    g_track = tf.Graph()
    with g_track.as_default() as graph:
        tf.import_graph_def(graph_def)
        track_sess = tf.Session(graph=graph)

        delta_move1 = track_sess.graph.get_tensor_by_name("import/delta_move1:0")
        cost_vol1 = track_sess.graph.get_tensor_by_name("import/cost_vol1:0")

        delta_move2 = track_sess.graph.get_tensor_by_name("import/delta_move2:0")
        cost_vol2 = track_sess.graph.get_tensor_by_name("import/cost_vol2:0")

        delta_move3 = track_sess.graph.get_tensor_by_name("import/delta_move3:0")
        cost_vol3 = track_sess.graph.get_tensor_by_name("import/cost_vol3:0")

        delta_move4 = track_sess.graph.get_tensor_by_name("import/delta_move4:0")
        Hmask_out = track_sess.graph.get_tensor_by_name("import/Hmask_out:0")

        input1 = track_sess.graph.get_tensor_by_name("import/input1:0")
        input2 = track_sess.graph.get_tensor_by_name("import/input2:0")


    track_in={
        'input1': input1,
        'input2': input2,
    }

    track_out={
        'delta_move1': delta_move1,
        'cost_vol1': cost_vol1,
        'delta_move2': delta_move2,
        'cost_vol2': cost_vol2,
        'delta_move3': delta_move3,
        'cost_vol3': cost_vol3,
        'delta_move4': delta_move4,
        'Hmask_out': Hmask_out,
    }

    return track_sess, track_in, track_out

def load_confidence_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    g_confidence = tf.Graph()
    with g_confidence.as_default() as graph:
        tf.import_graph_def(graph_def)
        confidence_sess = tf.Session(graph=graph)
        cost_volA = confidence_sess.graph.get_tensor_by_name("import/cost_volA:0")
        cost_volB = confidence_sess.graph.get_tensor_by_name("import/cost_volB:0")
        cost_volC = confidence_sess.graph.get_tensor_by_name("import/cost_volC:0")
        confidence = confidence_sess.graph.get_tensor_by_name("import/confidence:0")
        
    confidence_in={
        'cost_volA': cost_volA,
        'cost_volB': cost_volB,
        'cost_volC': cost_volC,
    }

    return confidence_sess, confidence_in, confidence

#################################################################################################################

def transfer(M_init2current, four_points):
    if(np.isnan(np.sum(four_points))):
        return four_points
    
    ONE = np.ones([four_points.shape[0],1])
    homogeneous = np.concatenate((four_points, ONE), axis=1)
    after_point = M_init2current.dot(homogeneous.T) #3, 4
    after_point = after_point.T #4, 3
    current_four_points = after_point[:, :2] / after_point[:, 2:]
    
    return current_four_points

def run_for_init(current_four_points, prvs_frame, current_frame, track_sess, track_in, track_out):
    
    if(np.isnan(np.sum(current_four_points))):
           return current_four_points
        
    height = 120
    width = 120
    height_v = prvs_frame.shape[0]
    width_v = prvs_frame.shape[1]

    template_point=np.array([(0,0),(width-1,0),(width-1,height-1),(0,height-1)])
    whole_img_four_points=np.array([(0,0),(width_v-1,0),(width_v-1,height_v-1),(0,height_v-1)])
    
    M_whole2template = cv2.getPerspectiveTransform( np.float32(whole_img_four_points), np.float32(template_point) )
    M_current_whole2template = cv2.getPerspectiveTransform( np.float32(whole_img_four_points), np.float32(template_point) )
    
    template_image = cv2.warpPerspective(prvs_frame, M_whole2template, (width, height))
    warp_image = cv2.warpPerspective(current_frame, M_current_whole2template, (width, height))

    loop_time = 3 #3-layer-pyramid
    for j in range(0,loop_time):

        patch1_batch = np.array([template_image])/ 127.5 - 1.0 
        patch2_batch = np.array([warp_image])/ 127.5 - 1.0 
        feed_dict = {track_in['input1']: patch1_batch, track_in['input2']: patch2_batch}

        if j==0:# lst level
            delta_move_np= track_sess.run(track_out['delta_move1'], feed_dict = feed_dict)
        if j==1: # 2nd level
            delta_move_np= track_sess.run(track_out['delta_move2'], feed_dict = feed_dict)
        if j==2: # 3rd level
            delta_move_np= track_sess.run(track_out['delta_move3'], feed_dict = feed_dict)

        vec2 = [0,0,width-1,0,width-1,height-1,0,height-1]
        vec1 = delta_move_np[0] + vec2
        vec1_point = [(vec1[0],vec1[1]),(vec1[2],vec1[3]),(vec1[4],vec1[5]),(vec1[6],vec1[7])]
        vec2_point = [(vec2[0],vec2[1]),(vec2[2],vec2[3]),(vec2[4],vec2[5]),(vec2[6],vec2[7])]
        M_delta = cv2.getPerspectiveTransform( np.float32(vec2_point), np.float32(vec1_point) )

        if np.linalg.det(M_current_whole2template) == 0:
            print("This matrix is singular, cannot be inversed!")
            M_pre2current_whole = M_current_whole2template 
        else:
            M_pre2current_whole = inv(M_current_whole2template)
        M_pre2current_whole = M_pre2current_whole.dot(M_delta)
        M_pre2current_whole = M_pre2current_whole.dot(M_whole2template)
        M_pre2current_whole = M_pre2current_whole/M_pre2current_whole[2,2]

        current_whole_four_points = transfer(M_pre2current_whole, whole_img_four_points)

        M_current_whole2template = cv2.getPerspectiveTransform( np.float32(current_whole_four_points), np.float32(template_point) )
        warp_image = cv2.warpPerspective(current_frame, M_current_whole2template, (width, height))

    current_four_points = transfer(M_pre2current_whole, current_four_points)
            
    return current_four_points


def run_pyramid(current_frame, 
                M_init2template, 
                template_image, 
                M_current2template, 
                init_four_points, 
                P_value, 
                track_sess, 
                track_in, 
                track_out, 
                confidence_sess, 
                confidence_in, 
                confidence):
    height = 120
    width = 120
    template_point=np.array([(0,0),(width-1,0),(width-1,height-1),(0,height-1)])
    warp_image = cv2.warpPerspective(current_frame, M_current2template, (width, height))

    loop_time = 3
    for j in range(0,loop_time):
        
        patch1_batch = np.array([template_image])/ 127.5 - 1.0 
        patch2_batch = np.array([warp_image])/ 127.5 - 1.0 
        feed_dict = {track_in['input1']: patch1_batch, track_in['input2']: patch2_batch}

        if j==0:# lst level
            delta_move_np, cost_vol_np1 = track_sess.run([track_out['delta_move1'], track_out['cost_vol1']], feed_dict = feed_dict)
        if j==1: # 2nd level
            delta_move_np, cost_vol_np2 = track_sess.run([track_out['delta_move2'], track_out['cost_vol2']], feed_dict = feed_dict)
        if j==2: # 3rd level
            delta_move_np, cost_vol_np3 = track_sess.run([track_out['delta_move3'], track_out['cost_vol3']], feed_dict = feed_dict)
            
        vec2 = [0,0,width-1,0,width-1,height-1,0,height-1]
        vec1 = delta_move_np[0]+ vec2
        vec1_point = [(vec1[0],vec1[1]),(vec1[2],vec1[3]),(vec1[4],vec1[5]),(vec1[6],vec1[7])]
        vec2_point = [(vec2[0],vec2[1]),(vec2[2],vec2[3]),(vec2[4],vec2[5]),(vec2[6],vec2[7])]
        M_delta = cv2.getPerspectiveTransform( np.float32(vec2_point), np.float32(vec1_point) )

        if np.linalg.det(M_current2template) == 0:
            print("This matrix is singular, cannot be inversed!")
            M_init2current = M_current2template 
        else:
            M_init2current = inv(M_current2template)
        M_init2current = M_init2current.dot(M_delta)
        M_init2current = M_init2current.dot(M_init2template)
        M_init2current = M_init2current/M_init2current[2,2]

        current_four_points = transfer(M_init2current, init_four_points)

        M_current2template = cv2.getPerspectiveTransform( np.float32(current_four_points), np.float32(template_point) )
        warp_image = cv2.warpPerspective(current_frame, M_current2template, (width, height))

    #cal_confidence
    feed_dict = {confidence_in['cost_volA']: cost_vol_np1, confidence_in['cost_volB']: cost_vol_np2, confidence_in['cost_volC']: cost_vol_np3}
    confidence_out = confidence_sess.run(confidence, feed_dict = feed_dict)

    return M_current2template, current_four_points, M_init2current, confidence_out >= P_value


def run_refinement(current_frame, M_init2template, template_image, M_current2template, loop_times, init_four_points, track_sess, track_in, track_out):

    height = 120
    width = 120
    template_point=np.array([(0,0),(width-1,0),(width-1,height-1),(0,height-1)])
    warp_image = cv2.warpPerspective(current_frame, M_current2template, (width, height))

    loop_time = loop_times
    for j in range(0,loop_time):
        patch1_batch = np.array([template_image])/ 127.5 - 1.0 
        patch2_batch = np.array([warp_image])/ 127.5 - 1.0 
        feed_dict = {track_in['input1']: patch1_batch, track_in['input2']: patch2_batch}

        delta_move_np = track_sess.run(track_out['delta_move4'], feed_dict = feed_dict)

        vec2 = [0,0,width-1,0,width-1,height-1,0,height-1]
        vec1 = delta_move_np[0]+ vec2
        vec1_point = [(vec1[0],vec1[1]),(vec1[2],vec1[3]),(vec1[4],vec1[5]),(vec1[6],vec1[7])]
        vec2_point = [(vec2[0],vec2[1]),(vec2[2],vec2[3]),(vec2[4],vec2[5]),(vec2[6],vec2[7])]
        M_delta = cv2.getPerspectiveTransform( np.float32(vec2_point), np.float32(vec1_point) )

        if np.linalg.det(M_current2template) == 0:
            print("This matrix is singular, cannot be inversed!")
            M_init2current = M_current2template 
        else:
            M_init2current = inv(M_current2template)
        M_init2current = M_init2current.dot(M_delta)
        M_init2current = M_init2current.dot(M_init2template)
        M_init2current = M_init2current/M_init2current[2,2]

        current_four_points = transfer(M_init2current, init_four_points)

        M_current2template = cv2.getPerspectiveTransform( np.float32(current_four_points), np.float32(template_point) )
        warp_image = cv2.warpPerspective(current_frame, M_current2template, (width, height))

    # vis mask of last step can be used for occlusion mask
    patch1_batch = np.array([template_image])/ 127.5 - 1.0 
    patch2_batch = np.array([warp_image])/ 127.5 - 1.0 
    feed_dict = {track_in['input1']: patch1_batch, track_in['input2']: patch2_batch}
    Hmask_out_np = track_sess.run(track_out['Hmask_out'], feed_dict = feed_dict)

    return M_current2template, current_four_points, M_init2current, Hmask_out_np


def H_to_tml_use_sift_reinit(template_image, frame):
    img1_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  
    img2_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)  
    
    sift = cv2.xfeatures2d.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1_gray, None)
    kp2, des2 = sift.detectAndCompute(img2_gray, None)

    # BFmatcher with default parms
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des1, des2, k = 2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance: ##0.7
            good.append(m)

    MIN_MATCH_COUNT = 8
    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)    
    else:
        M = None
    return M

def update_float_template(current_frame, current_four_points, template_point_HVC, width_HVC, height_HVC):
    M_current2template = cv2.getPerspectiveTransform( np.float32(current_four_points), np.float32(template_point_HVC) )
    float_template = cv2.warpPerspective(current_frame, M_current2template, (width_HVC, height_HVC))
    float_four_points = current_four_points.copy()
    M_float2template = M_current2template.copy()

    return float_template, float_four_points, M_float2template

def run_HVC(current_four_points, 
            template_point_HVC, 
            current_frame, 
            M_init2template, 
            template_image_HVC, 
            init_four_points, thr, 
            track_sess, track_in, track_out,
            confidence_sess, confidence_in, confidence
            ):
    M_current2template = cv2.getPerspectiveTransform( np.float32(current_four_points), np.float32(template_point_HVC) )
    M_current2template, current_four_points, M_init2current, confidence = run_pyramid(current_frame, 
                                                                                        M_init2template, 
                                                                                        template_image_HVC, 
                                                                                        M_current2template, 
                                                                                        init_four_points, 
                                                                                        thr, 
                                                                                        track_sess, track_in, track_out, 
                                                                                        confidence_sess, confidence_in, confidence)
    M_current2template, current_four_points, M_init2current, pre_mask = run_refinement(current_frame, M_init2template, template_image_HVC, M_current2template, 3, init_four_points, track_sess, track_in, track_out)


    return M_current2template, current_four_points, M_init2current, confidence, pre_mask


def track(track_pb, confidence_pb, V_path, Vinitpoint_path, out_V_path, logo_path):

    track_sess, track_in, track_out = load_tracking_graph(track_pb)
    confidence_sess, confidence_in, confidence = load_confidence_graph(confidence_pb)

    logo = cv2.resize(cv2.imread(logo_path), (120,120))

    cap=cv2.VideoCapture(V_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) ) 
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    v_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(out_V_path, fourcc, fps, (width*2,height))

    ## load tracked planar object
    init_four_points = (np.loadtxt(Vinitpoint_path)).reshape([4,2])
    ret, frame = cap.read()

    pic = frame.copy()
    cv2.polylines(pic, np.int32(np.round([init_four_points])), 1, (0,255,0), 2)
    out_frame = np.hstack((pic, pic))
    out.write(np.uint8(out_frame))

    ############################### HVC-tml
    height_HVC = 120
    width_HVC= 120
    template_point_HVC=[(0,0),(width_HVC-1,0),(width_HVC-1,height_HVC-1),(0,height_HVC-1)]#(x,y)
    M_init2template = cv2.getPerspectiveTransform( np.float32(init_four_points), np.float32(template_point_HVC) )
    template_image_HVC = cv2.warpPerspective(frame, M_init2template, (width_HVC, height_HVC))  # width , height

    ############################### sift-tml
    height_sift = int(init_four_points[2][1] - init_four_points[1][1])
    width_sift = int(init_four_points[1][0] - init_four_points[0][0])
    template_point_sift=[(0,0),(width_sift-1,0),(width_sift-1,height_sift-1),(0,height_sift-1)]#(x,y)
    M_init2template_sift = cv2.getPerspectiveTransform( np.float32(init_four_points), np.float32(template_point_sift) )
    template_image_sift = cv2.warpPerspective(frame, M_init2template_sift, (width_sift, height_sift))  # width , height

    prvs_frame = frame
    current_four_points = init_four_points      
    pre_mask = np.zeros([1,120,120])

    float_template, float_four_points, M_float2template = update_float_template(frame, current_four_points, template_point_HVC, width_HVC, height_HVC)

    #--------------------------------start-------------------------------------#
    for i in range(1,v_length):
        if(i%50==0):
            print(i, v_length)

        use_sift_reinit = 0 
        confidence_out = False

        ret, current_frame = cap.read()
        last_four_points = current_four_points.copy()
        
        # try w camera motion init
        current_four_points = run_for_init(current_four_points, prvs_frame, current_frame, track_sess, track_in, track_out)
        M_current2template, current_four_points, M_init2current, confidence_out, pre_mask = run_HVC(current_four_points, 
                                                                                                template_point_HVC, 
                                                                                                current_frame, 
                                                                                                M_init2template, 
                                                                                                template_image_HVC, 
                                                                                                init_four_points, 0.1,
                                                                                                track_sess, track_in, track_out,
                                                                                                confidence_sess, confidence_in, confidence
                                                                                                )

        #update_float_template
        OCC_range_temp = np.mean(pre_mask)  
        if(confidence_out & (OCC_range_temp<0.2)):
            float_template_tmp, float_four_points_tmp, M_float2template_tmp = update_float_template(frame, current_four_points, template_point_HVC, width_HVC, height_HVC)
            if (np.mean(float_template_tmp==0)<0.05):
                float_template, float_four_points, M_float2template = float_template_tmp.copy(), float_four_points_tmp.copy(), M_float2template_tmp.copy()

        if(~confidence_out):
            #try using float_template
            current_four_points = last_four_points.copy()
            current_four_points = run_for_init(current_four_points, prvs_frame, current_frame, track_sess, track_in, track_out)
            M_current2template, current_four_points, M_init2current, confidence_out, pre_mask = run_HVC(current_four_points, 
                                                                                                    template_point_HVC, 
                                                                                                    current_frame, 
                                                                                                    M_float2template, 
                                                                                                    float_template, 
                                                                                                    float_four_points, 0.5, 
                                                                                                    track_sess, track_in, track_out,
                                                                                                    confidence_sess, confidence_in, confidence
                                                                                                    )
            
            if (~confidence_out):
                #try w/o camera motion init
                current_four_points = last_four_points.copy()
                M_current2template, current_four_points, M_init2current, confidence_out, pre_mask = run_HVC(current_four_points, 
                                                                                                    template_point_HVC, 
                                                                                                    current_frame, 
                                                                                                    M_init2template, 
                                                                                                    template_image_HVC, 
                                                                                                    init_four_points, 0.1, 
                                                                                                    track_sess, track_in, track_out,
                                                                                                    confidence_sess, confidence_in, confidence
                                                                                                    )

                #update_float_template
                OCC_range_temp = np.mean(pre_mask)  
                if(confidence_out & (OCC_range_temp<0.2)):
                    float_template_tmp, float_four_points_tmp, M_float2template_tmp = update_float_template(frame, current_four_points, template_point_HVC, width_HVC, height_HVC)
                    if (np.mean(float_template_tmp==0)<0.05):
                        float_template, float_four_points, M_float2template = float_template_tmp.copy(), float_four_points_tmp.copy(), M_float2template_tmp.copy()

                if(~confidence_out):   
                    #try using float_template
                    current_four_points = last_four_points.copy()
                    M_current2template, current_four_points, M_init2current, confidence_out, pre_mask = run_HVC(current_four_points, 
                                                                                                            template_point_HVC, 
                                                                                                            current_frame, 
                                                                                                            M_float2template, 
                                                                                                            float_template, 
                                                                                                            float_four_points, 0.5, 
                                                                                                            track_sess, track_in, track_out,
                                                                                                            confidence_sess, confidence_in, confidence
                                                                                                            )

                    #need to use sift to re-init
                    if (~confidence_out): 
                        use_sift_reinit = 1
        
        # use_sift_reinit  
        if(use_sift_reinit == 1):    
            current_four_points = last_four_points.copy()        
            H_current_frame_to_tml = H_to_tml_use_sift_reinit(template_image_sift, current_frame)
            if ((H_current_frame_to_tml is not None)):
                M_init2current = (inv(H_current_frame_to_tml)).dot(M_init2template_sift)
                four_points_guess = transfer(M_init2current, init_four_points)
                M_current2template = cv2.getPerspectiveTransform( np.float32(four_points_guess), np.float32(template_point_HVC) )
                _, _, _, pre_mask = run_refinement(current_frame, M_init2template, template_image_HVC, M_current2template, 1, init_four_points, track_sess, track_in, track_out)
                OCC_range_temp = np.mean(pre_mask)
                if (OCC_range_temp<0.8):
                    current_four_points = four_points_guess.copy()      
                    print(i, 'use sift re-init')     
                else:
                    print(i, 'use last') 
            else:
                    print(i, 'use last')                       
                
        logo_tmp = cv2.warpPerspective(logo, inv(M_current2template), (width, height))
        occ_mask = cv2.warpPerspective(1.0-pre_mask[0].astype(np.float32), inv(M_current2template), (width, height))
        occ_mask = np.tile(occ_mask[..., np.newaxis], [1,1,3])
        
        pic = current_frame.copy()  
        cv2.polylines(pic, np.int32(np.round([current_four_points])), 1, (0,255,0), 2)
        pic2 = pic * (1.0 - occ_mask)  +  logo_tmp * occ_mask
        out_frame = np.hstack((pic, pic2))
        out.write(np.uint8(out_frame))
            
        prvs_frame = current_frame

    cap.release()
    out.release()

    print('finish')
    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--track_pb', type=str, default='HVC_tracking_part_fast_version.pb', help='track_pb path')
    parser.add_argument('--confidence_pb', type=str, default='HVC_confidence_part.pb', help='confidence_pb path')
    parser.add_argument('--V_path', type=str, default='./test_data/V15_7.avi', help='input_video_path')
    parser.add_argument('--Vinitpoint_path', type=str, default='./test_data/V15_7_init_points.txt', help='input_init_points')
    parser.add_argument('--out_V_path', type=str, default='./results/V15_7_out.mp4', help='output_video_path')
    parser.add_argument('--logo_path', type=str, default='./test_data/CVF.png', help='logo')
    opt = parser.parse_args()

    track(opt.track_pb, opt.confidence_pb, opt.V_path, opt.Vinitpoint_path, opt.out_V_path, opt.logo_path)