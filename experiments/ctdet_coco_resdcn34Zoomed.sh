cd src
# train
python main.py ctdet --exp_id coco_res2dcnZoomed18 --arch res2dcnZoomed_18 --batch_size 1 --master_batch 1 --lr 3.75e-4
# test
python test.py ctdet --exp_id coco_res2dcnZoomed18 --arch res2dcnZoomed_18 --keep_res --resume
# flip test
python test.py ctdet --exp_id coco_res2dcnZoomed18 --arch res2dcnZoomed_18 --keep_res --resume --flip_test
# multi scale test
python test.py ctdet --exp_id coco_res2dcnZoomed18 --arch res2dcnZoomed_18 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5

cd ..
