key_num=8
save_path=../Results_davis/
#    --show \ # see visual results
python3 tools/test_video_seq_davis.py \
    configs/SeqMaskRCNN/seq_mask_rcnn_r101_fpn_1x_davisuvos_ftstage.py \
    pretrained/seq_mask_rcnn_r101_davis.pth \
    --gpu '0' \
    --network r101_Ft \
    --key-num $key_num \
    --out $save_path/save.pkl \
    --eval segm \
    --save-dir $save_path \
