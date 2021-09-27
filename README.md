# Video Instance Segmentation with a Propose-Reduce Paradigm ([ICCV 2021](https://arxiv.org/abs/2103.13746))
 
![TeaserImage](https://github.com/HuaijiaLin/ProposeReduce-debug/blob/main/.images/teaser.gif)
 
## Installation
Please refer to [INSTALL.md](INSTALL.md).

## Demo
You can compute the VIS results for your own videos.
1. Download [pretrained weight](https://github.com/HuaijiaLin/ProposeReduce-debug/blob/main/README.md#results).
2. Put example videos in 'demo/inputs'. We support two types of inputs, *frames* directories or *.mp4* files (see [example](https://github.com/HuaijiaLin/ProposeReduce-debug/tree/main/demo/inputs) for details).
3. Run the following script and obtain the results in 'demo/outputs'.
```
sh demo.sh
```

## Data Preparation
(1) Download the videos and jsons of *val* set from [YouTube-VIS 2019](https://competitions.codalab.org/competitions/20128#participate-get-data)

(2) Symlink the corresponding dataset and json files to the 'data' folder
```
data
├── valset_ytv19 --> /path/to/ytv2019/vos/valid/JPEGImages/ 
├── valid_ytv19.json --> /path/to/ytv2019/vis/JSONAnnt/valid.json
```

## Results
We provide the results of several pretrained models and corresponding scripts on different backbones.

Download the pretrained models and put them in 'pretrained' folder.

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="center">Dataset</th>
<th valign="center">Method</th>
<th valign="center">Backbone</th>
<th valign="center">AP</th>
<th valign="center">AR@10</th>
<th valign="bottom">download</th>
  
<tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-50</td>
<td align="center"> 40.8 </td>
<td align="center"> 49.9 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1P3HiwCavjRJJePuF-4D2GDQKwWT8E_LZ/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/HuaijiaLin/ProposeReduce-debug/blob/main/scripts/YTV2019/eval_vis_r50.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
  
<tr><tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNet-101</td>
<td align="center"> 43.8 </td>
<td align="center"> 52.7 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1SmcJsIqluzjuH-uKCNs1ybNqvQClIqai/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/HuaijiaLin/ProposeReduce-debug/blob/main/scripts/YTV2019/eval_vis_r101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->
  
<tr><tr><td align="center">YouTube-VIS 2019</td>
<td align="center">Seq Mask R-CNN</td>
<td align="center">ResNeXt-101</td>
<td align="center"> 47.6 </td>
<td align="center"> 56.7 </td>
<td align="center"> <a href="https://drive.google.com/file/d/1lwjdGhjeA8rFtHtYrJbsVPY6r49jGGbN/view?usp=sharing">model</a>&nbsp;|&nbsp;<a href="https://github.com/HuaijiaLin/ProposeReduce-debug/blob/main/scripts/YTV2019/eval_vis_x101.sh">scripts</a> </td>
<!-- <td align="center"> To be released </td> -->

</tbody></table>

### Evaluation
**YouTube-VIS 2019**: A json file will be saved in `../Results' folder. Please zip and upload to the [codalab server](https://competitions.codalab.org/competitions/20128#participate-submit_results).

## TODOs
----------------
  - [ ] Results on YouTube-VIS 2021
  - [ ] Results on DAVIS-UVOS
  - [ ] [Category-Aware Sequence Reduction](https://youtube-vos.org/assets/challenge/2021/reports/VIS_4_Lin.pdf)
  - [ ] Training Codes

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
Parts of this code were derived from [MMDetection](https://github.com/open-mmlab/mmdetection), [MaskTrackRCNN](https://github.com/youtubevos/MaskTrackRCNN), [STM](https://github.com/seoungwugoh/STM), [MMCV](https://github.com/open-mmlab/mmcv) and [COCOAPI](https://github.com/youtubevos/cocoapi).
