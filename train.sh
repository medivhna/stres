CUDA_VISIBLE_DEVICES=0,1 \
python python/resnet_train.py \
--batch_size=128 \
--num_gpus=2 \
--lr_decay_epoches=4.0 \
--num_examples=4598268 \
--num_classes=93643 \
--train_list_path='/home/wangguanshuo/lists/MSRA/train.npy' \
--train_dir='/home/wangguanshuo/tf/model/resnet_50' \
--summary_dir='/home/wangguanshuo/tf/summary/resnet_50' \

# For more param changes in resnet_train.py
