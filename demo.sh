ep=4
key_num=5
name=test_masktrackrcnn_FullPre_FullTrain
#    --use-img # blending img and mask
python3 tools/demo.py \
    configs/SeqMaskRCNN/seq_mask_rcnn_r101_fpn_1x_youtubevos19_ftstage.py \
    pretrained/seq_mask_rcnn_r101_ytv19.pth \
    --gpu '0' \
    --network r50_Ft \
    --key-num $key_num \
    --eval segm \
    --input-dir demo/inputs \
    --save-dir demo/outputs \
