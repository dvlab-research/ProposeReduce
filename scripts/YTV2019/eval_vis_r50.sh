key_num=5
name=test_masktrackrcnn_FullPre_FullTrain
save_path=../Results_ytv19
#    --show \ # see visual results
python3 tools/test_video_seq.py \
    configs/SeqMaskRCNN/seq_mask_rcnn_r50_fpn_1x_youtubevos19_ftstage.py \
    pretrained/seq_mask_rcnn_r50_ytv19.pth \
    --gpu '0' \
    --network r50_Ft \
    --key-num $key_num \
    --out $save_path/save.pkl \
    --eval segm \
    --save-dir $save_path \
