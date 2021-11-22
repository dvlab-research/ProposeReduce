# Propose-Reduce VIS
This repo contains the official implementation for the paper:

**Video Instance Segmentation with a Propose-Reduce Paradigm**

*Huaijia Lin\*, Ruizheng Wu\*, Shu Liu, Jiangbo Lu, Jiaya Jia*

ICCV 2021 | [Paper](https://arxiv.org/abs/2103.13746) 
 
![TeaserImage](https://github.com/dvlab-research/ProposeReduce/blob/main/.images/teaser.gif)
 
## Installation
Please refer to [INSTALL.md](INSTALL.md).

## Demo
You can compute the VIS results for your own videos.
1. Download a pretrained [ResNet-101](https://drive.google.com/file/d/1SmcJsIqluzjuH-uKCNs1ybNqvQClIqai/view?usp=sharing) and put it in ***pretrained*** folder.
```
mkdir pretrained
```
2. Put example videos in 'demo/inputs'. We support two types of inputs, *frames* directories or *.mp4* files (see [example](https://github.com/dvlab-research/ProposeReduce/tree/main/demo/inputs) for details).
3. Run the following script and obtain the results in ***demo/outputs***.
```
sh demo.sh
```

## Data Preparation
(1) Download the videos and jsons of *train* and *val* sets from [YouTube-VIS 2019](https://competitions.codalab.org/competitions/20128#participate-get-data)

(2) Download the videos and jsons of *train* and *val* sets from [YouTube-VIS 2021](https://competitions.codalab.org/competitions/28988#participate-get_data)

(3) Download the *trainval* set of [DAVIS-UVOS](https://davischallenge.org/davis2017/code.html)

(4) Download other pre-computed jsons from [data](https://drive.google.com/drive/folders/1E0xpD6DwWwFzUUIo9dgG7T9-OlqDDOKs?usp=sharing)

(5) Symlink the corresponding dataset and json files to the ***data*** folder
```
mkdir data
```
```
data
├── trainset_ytv19 --> /path/to/ytv2019/vos/train/JPEGImages/
├── train_ytv19.json --> /path/to/ytv2019/vis/train.json
├── valset_ytv19 --> /path/to/ytv2019/vos/valid/JPEGImages/
├── valid_ytv19.json --> /path/to/ytv2019/vis/valid.json
├── trainset_ytv21 --> /path/to/ytv2021/vis/train/JPEGImages/ 
├── train_ytv21.json --> /path/to/ytv2021/vis/train/instances.json
├── valset_ytv21 --> /path/to/ytv2021/vis/valid/JPEGImages/ 
├── valid_ytv21.json --> /path/to/ytv2021/vis/valid/instances.json
├── trainvalset_davis --> /path/to/DAVIS-UnVOS/DAVIS-trainval/JPEGImages/480p/ 
├── train_davis.json --> /path/to/pre-computed/train_davis.json
├── valid_davis.json --> /path/to/pre-computed/valid_davis.json
```

## Results
We provide the results of several pretrained models and corresponding scripts on different backbones.
The results have slight differences from the paper because we make minor modifications to the inference codes.

Download the pretrained models and put them in ***pretrained*** folder.
```
mkdir pretrained
```

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Method</th>
<th valign="center">Backbone</th>
 <th valign="center"> <a href=https://github.com/dvlab-research/ProposeReduce#todos>CA Reduce</a> </th>
<th valign="center">AP</th>
<th valign="center">AR@10</th>
<th valign="bottom">download</th>
  
<tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-50</td>
<td align="center"></td>
<td align="center"> 40.8 </td>
<td align="center"> 49.9 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1P3HiwCavjRJJePuF-4D2GDQKwWT8E_LZ/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2019/eval_vis_r50.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-50</td>
<td align="center"> &check; </td>
<td align="center"> 42.5 </td>
<td align="center"> 56.8 </td>
<td align="center"> <a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2019/CateAwareReduce/eval_vis_r50.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
  
<tr><tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-101</td>
<td align="center"></td>
<td align="center"> 43.8 </td>
<td align="center"> 52.7 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1SmcJsIqluzjuH-uKCNs1ybNqvQClIqai/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2019/eval_vis_r101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-101</td>
<td align="center"> &check; </td>
<td align="center"> 45.2 </td>
<td align="center"> 59.0 </td>
<td align="center"> <a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2019/CateAwareReduce/eval_vis_r101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
  
<tr><tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNeXt-101</td>
<td align="center"></td>
<td align="center"> 47.6 </td>
<td align="center"> 56.7 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1lwjdGhjeA8rFtHtYrJbsVPY6r49jGGbN/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2019/eval_vis_x101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNeXt-101</td>
<td align="center"> &check; </td> 
<td align="center"> 48.8 </td>
<td align="center"> 62.2 </td>
<td align="center"> <a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2019/CateAwareReduce/eval_vis_x101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td> 
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<!-- <td align="center"> To be released </td> -->
 
<tr><td align="center">YouTube-VIS 2021</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-50</td>
<td align="center"></td>  
<td align="center"> 39.6 </td>
<td align="center"> 47.5 </td>
<td align="center"> <a href="https://drive.google.com/file/d/12NQMY59USqMi7--zyZytKVaUmf0MGegP/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2021/eval_vis_r50.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><td align="center">YouTube-VIS 2021</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-50</td>
<td align="center"> &check; </td>  
<td align="center"> 41.7 </td>
<td align="center"> 54.9 </td>
<td align="center"> <a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2021/CateAwareReduce/eval_vis_r50.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center">YouTube-VIS 2021</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNeXt-101</td>
<td align="center"> </td>  
<td align="center"> 45.6 </td>
<td align="center"> 52.9 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1aOHPmVkoF9ZeBOSORlybPBqpZoIqg2SA/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2021/eval_vis_x101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center">YouTube-VIS 2021</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNeXt-101</td>
<td align="center"> &check; </td>  
<td align="center"> 47.2 </td>
<td align="center"> 57.6 </td>
<td align="center"> <a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/YTV2021/CateAwareReduce/eval_vis_x101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->

</tbody></table>

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Method</th>
<th valign="center">Backbone</th>
<th valign="center">J&F</th>
<th valign="center">J</th>
<th valign="center">F</th>
<th valign="bottom">download</th>
 
<tr><tr><td align="center">DAVIS-UVOS</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-101</td>
<td align="center"> 68.1 </td>  
<td align="center"> 64.9 </td>
<td align="center"> 71.4 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1gOgpEQ1rhFVCRRqR98Jr4s9MhWMUPvzl/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/DAVIS/eval_vis_r101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center">DAVIS-UVOS</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNeXt-101</td>
<td align="center"> 70.6 </td>  
<td align="center"> 67.2 </td>
<td align="center"> 73.9 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1fKNCS2ONTD3q9B4oB8TCTpMz7J0CLNtX/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/main/scripts/DAVIS/eval_vis_x101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
 </tbody></table>

### Evaluation
**YouTube-VIS 2019**: A json file will be saved in ***../Results_ytv19*** folder. Please zip and upload to the [codalab server](https://competitions.codalab.org/competitions/20128#participate-submit_results).

**YouTube-VIS 2021**: A json file will be saved in ***../Results_ytv21*** folder. Please zip and upload to the [codalab server](https://competitions.codalab.org/competitions/28988#participate-submit_results).

**DAVIS-UVOS**: Color masks will be saved in ***../Results_davis*** folder. Please use the [official code](https://github.com/davisvideochallenge/davis2017-evaluation#evaluate-davis-2017-unsupervised) for evaluation.

## Training
To reproduce the results, we provide the pre-trained model on the **main-training stage** and the training scripts for the **finetuning stage** (described in Sec. 4.2 of the paper).

Please put the pre-trained model into ***pretrained*** folder and then run the corresponding script.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Method</th>
<th valign="center">Backbone</th>
<th valign="bottom">download</th>
  
<tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-50</td>
<td align="center"> <a href="https://drive.google.com/file/d/15GpsH2Owgv57yruLEUM1kuotOW0xFO3q/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/train/scripts/YTV2019/run_vis_r50.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
   
<tr><tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-101</td>
<td align="center"> <a href="https://drive.google.com/file/d/1XjqdryRhsFsYWH1m2O1TQGhbfwv57rZV/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/train/scripts/YTV2019/run_vis_r101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
   
<tr><tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNeXt-101</td>
<td align="center"> <a href="https://drive.google.com/file/d/17NNQcvpYPKEV-P7RswhbuxxBIa0cwAYe/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/train/scripts/YTV2019/run_vis_x101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
  
<tr><tr><td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<!-- <td align="center"> To be released </td> -->
 
<tr><td align="center">YouTube-VIS 2021</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-50</td>
<td align="center"> <a href="https://drive.google.com/file/d/1vn15yt_j27wz0eIpoYy3IoacRbdhMiLm/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/train/scripts/YTV2021/run_vis_r50.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center">YouTube-VIS 2021</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNeXt-101</td>
<td align="center"> <a href="https://drive.google.com/file/d/1IxGHJ77xK4c6f_3SsyCQnwG6obRhBPHz/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/train/scripts/YTV2021/run_vis_x101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<td align="center"></td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center">DAVIS-UVOS</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-101</td>
<td align="center"> <a href="https://drive.google.com/file/d/1NqyuMWbORNIYWhR6duk3Xw94w8yKxBPK/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/train/scripts/DAVIS/run_vis_r101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
<tr><tr><td align="center">DAVIS-UVOS</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNeXt-101</td>
<td align="center"> <a href="https://drive.google.com/file/d/1EojHV0cmHV349ryPo3RGLS2Y_7lnoSUa/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/dvlab-research/ProposeReduce/blob/train/scripts/DAVIS/run_vis_x101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
 
</tbody></table>

The trained checkpoints will be saved in ***../work_dirs*** folder. To evaluate the effect, please replace the pretrained weights of [inference](https://github.com/dvlab-research/ProposeReduce/tree/train#results) with the trained checkpoints and run the inference scripts.

## TODOs
  - [x] Results on YouTube-VIS 2021
  - [x] Results on DAVIS-UVOS
  - [x] [Category-Aware Sequence Reduction (CA Reduce)](https://youtube-vos.org/assets/challenge/2021/reports/VIS_4_Lin.pdf)
  - [x] Training Codes

## Citation
If you find this work useful in your research, please cite:
```
@article{lin2021video,
  title={Video Instance Segmentation with a Propose-Reduce Paradigm},
  author={Lin, Huaijia and Wu, Ruizheng and Liu, Shu and Lu, Jiangbo and Jia, Jiaya},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  year={2021}
}
```

## Contact
If you have any questions regarding the repo, please feel free to contact me (huaijialin@gmail.com) or create an issue.

## Acknowledgments
This repo is based on [MMDetection](https://github.com/open-mmlab/mmdetection), [MaskTrackRCNN](https://github.com/youtubevos/MaskTrackRCNN), [STM](https://github.com/seoungwugoh/STM), [MMCV](https://github.com/open-mmlab/mmcv) and [COCOAPI](https://github.com/youtubevos/cocoapi).
