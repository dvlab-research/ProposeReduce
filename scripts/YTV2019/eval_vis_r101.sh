key_num=5
name=test_masktrackrcnn_FullPre_FullTrain
save_path=../Results
#    --show \ # see visual results
python3 tools/test_video_v1.py \
    configs/SeqMaskRCNN/seq_mask_rcnn_r101_fpn_1x_youtubevos19_ftstage.py \
    pretrained/seq_mask_rcnn_r101_ytv19.pth \
    --gpu '0' \
    --network r101_Ft \
    --key-num $key_num \
    --out $save_path/save.pkl \
    --eval segm \
    --save-dir $save_path \
