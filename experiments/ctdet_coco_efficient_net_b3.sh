cd src
# train
python main.py ctdet --exp_id efficient_net_b3 --arch efficientNetb_3 --batch_size 120 --master_batch 5  --lr 4.6875e-4 --gpus 0,1,2,3,4,5,6,7 --num_workers 16 --head_conv 256
# test
python test.py ctdet --exp_id efficient_net_b3 --arch efficientNetb_3 --keep_res --resume
# flip test
python test.py ctdet --exp_id efficient_net_b3 --arch efficientNetb_3 --keep_res --resume --flip_test
# multi scale test
python test.py ctdet --exp_id efficient_net_b3 --arch efficientNetb_3 --keep_res --resume --flip_test --test_scales 0.5,0.75,1,1.25,1.5

cd ..
