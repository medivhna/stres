CUDA_VISIBLE_DEVICES=0,1,2,3 \
python python/resnet_train.py \
--batch_size=128 \
--num_gpus=4 \
--lr_decay_epoches=4.0 \
--num_examples=4600023 \
--num_classes=93644 \
--train_list_path='/home/wangguanshuo/lists/cropped/stn_cropped_with_params.npy' \
--train_dir='/home/wangguanshuo/tf/model/stres' \
--summary_dir='/home/wangguanshuo/tf/summary/stres' \

# For more param changes in resnet_train.py
