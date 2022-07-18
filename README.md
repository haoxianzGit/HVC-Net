# HVC-Net: Unifying Homography, Visibility, and Confidence Learning for Planar Object Tracking (ECCV 2022)



<!-- <video src="videos/video_demo.mp4" controls="controls" width="700"></video> --> 

<img src="figures/video_demo.gif" width="700" />

<br>

This repository provides the test code of the following paper:

> **HVC-Net: Unifying Homography, Visibility, and Confidence Learning for Planar Object Tracking (ECCV 2022)**.\
> Haoxian Zhang*, Yonggen Ling*. (*Equal contribution) \
> paper link: *coming soon* 

> **Abstract**: *Robust and accurate planar tracking over a whole video sequence is vitally important for many vision applications. The key to planar object tracking is to find object correspondences, modeled by homography, between the reference image and the tracked image. Existing methods tend to obtain wrong correspondences with changing appearance variations, camera-object relative motions and occlusions. To alleviate this problem, we present a unified convolutional neural network (CNN) model that jointly considers homography, visibility, and confidence. First, we introduce correlation blocks that explicitly account for the local appearance changes and camera-object relative motions as the base of our model. Second, we jointly learn the homography and visibility that links camera-object relative motions with occlusions. Third, we propose a confidence module that actively monitors the estimation quality from the pixel correlation distributions obtained in correlation blocks. All these modules are plugged into a Lucas-Kanade (LK) tracking pipeline to obtain both accurate and robust planar object tracking. Our approach outperforms the state-of-the-art methods on public POT and TMT datasets. Its superior performance is also verified on a real-world application, synthesizing high-quality in-video advertisements.*



## Dependencies

- python 3.6
- tensorflow 1.8 
- opencv
- numpy


## How to run ?

Download our **model files** and unzip it to the **./model** floder: [Google Drive](https://drive.google.com/file/d/1ZgW5KDKZOfbyH0p8yT3HzLL0SXW237Xl/view?usp=sharing) or  [腾讯微云](https://share.weiyun.com/xCM0jF2D).

Download **test data** and unzip it to the **./test_data** floder: [Google Drive](https://drive.google.com/file/d/1ipB-hXEllSowV-6jANVr1KC_ImcKfR-W/view?usp=sharing) or  [腾讯微云](https://share.weiyun.com/xCM0jF2D).

Please run the follow command to generate the results.

```
python run.py \
  --track_pb='model/HVC_tracking_part_fast_version.pb' \
  --confidence_pb='model/HVC_confidence_part.pb' \
  --V_path='./test_data/V15_7.avi' \
  --Vinitpoint_path='./test_data/V15_7_init_points.txt' \
  --out_V_path='./results/V15_7_out.mp4' \
```
- track_pb :  HVC_tracking_part.pb / HVC_tracking_part_fast_version.pb 
- confidence_pb: HVC_confidence_part.pb 
- V_path : input video path
- Vinitpoint_path: You need to mark the coordinates of the tracked planar object in the first frame.
- out_V_path : output video path


## Results

<!-- <video src="results/V15_1_out.mp4" controls="controls" width="700"></video> --> 
<!-- <video src="results/V15_7_out.mp4" controls="controls" width="700"></video> --> 

<img src="figures/V15_1_out.gif" width="700" />
<img src="figures/V15_7_out.gif" width="700" />

## License and Citation

```
The provided implementation is for academic purposes only. If you use the code provided in this repository, please cite our paper as follows.

@article{Zhang2022HVC,
  title={HVC-Net: Unifying Homography, Visibility, and Confidence Learning for Planar Object Tracking},
  author={Zhang, Haoxian and Ling, Yonggen},
  booktitle ={European Conference on Computer Vision},
  year={2022}
}
```



