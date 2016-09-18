CUDA_VISIBLE_DEVICES=0 \
python python/resnet_extract.py \
--batch_size=200 \
--num_gpus=1 \
--num_examples=545913 \
--delimeter=' ' \
--checkpoint_path='/home/wangguanshuo/resnet/resnet_model/resnet_50' \
--extract_list_path='/home/wangguanshuo/lists/youtube/youtube_aligned.npy' \
--fea_dir='/home/wangguanshuo/fea/resnet_50' \
--fea_file='fea_ytb.mat' \

# For more param changes in resnet_train.py
