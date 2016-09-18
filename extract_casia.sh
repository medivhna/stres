CUDA_VISIBLE_DEVICES=1 \
python python/resnet_extract.py \
--batch_size=200 \
--num_gpus=1 \
--num_examples=516046 \
--delimeter=' ' \
--checkpoint_path='/home/wangguanshuo/resnet/resnet_model/resnet_50' \
--extract_list_path='/home/wangguanshuo/lists/face/training_face.npy' \
--fea_dir='/home/wangguanshuo/fea/resnet_50_2' \
--fea_file='fea_casia.mat' \

# For more param changes in resnet_train.py
