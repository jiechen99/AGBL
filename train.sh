python2 multigpu_train.py --gpu_list=0 --input_size=512 --batch_size_per_gpu=8 --checkpoint_path=./curve_model/ --text_scale=512 --training_data_path=/media/icstcscl/data-950G/dataset/ctw1500/train/ctw_train1000/  --geometry=RBOX --learning_rate=0.001 --num_readers=24 --pretrained_model_path=/home/icstcscl/resnet_v1_50.ckpt

