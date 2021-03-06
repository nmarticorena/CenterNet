cd src
# train
python main.py ctdet --exp_id coco_res2dcnZoomed18 --arch res2dcnZoomed_18 --resume --batch_size 228 --master_batch 18 --lr 1e-3 --gpus 0,1,2,3,4,5,6,7 --num_workers 16
# test
python test.py ctdet --exp_id coco_res2dcnZoomed18 --arch res2dcnZoomed_18 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_res2dcnZoomed18 --arch res2dcnZoomed_18 --keep_res --resume --flip_test
# multi scale test
python test.py ctdet --exp_id coco_res2dcnZoomed18 --arch res2dcnZoomed_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5

cd ..
