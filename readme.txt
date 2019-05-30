准备数据：
视频，组织方式为：
输入视频的文件夹:
	\每个人的一个文件夹\这个人的视频
	\每个人的一个文件夹\这个人的视频

配置：Openface， dlib

人脸检测、裁剪、生成mask的命令
python3 video_frame.py 输入视频的文件夹 保存到的文件夹 --image_size 256 --scale 1.2

生成AU，需要使用OpenFace
/home/mean/demo/OpenFace/build/bin/FeatureExtraction -fdir 输入图片上路径所在的文件夹 -aus -out_dir 输出文件夹
会得到两个文件，一个csv，一个txt，主要用csv就行

把au的csv文件转化为npy文件，使用prepare_au_annotations.py
python3 prepare_au_annotations.py -ia 输入包含csv的文件夹 -op 输出保存路径

把所有的图片，类重新保存，用prepare_ids_annotations.py，这个文件把所有的文件路径保存在一个新的csv文件
同时生成npy文件，npy文件的第i行更csv文件第i行对应的图片对应。这个不一定需要用。
