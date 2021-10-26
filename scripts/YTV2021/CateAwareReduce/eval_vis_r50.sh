key_num=5
save_path=../Results_ytv21
#    --show \ # see visual results
python3 tools/test_video_seq.py \
    configs/SeqMaskRCNN/CateAwareReduce/seq_mask_rcnn_r50_fpn_1x_youtubevis21_ftstage.py \
    pretrained/seq_mask_rcnn_r50_ytv21.pth \
    --gpu '0' \
    --network r50_Ft_CAReduce \
    --key-num $key_num \
    --out $save_path/save.pkl \
    --eval segm \
    --save-dir $save_path \
    --use-cate-reduce \
    --score-thr 0.001
