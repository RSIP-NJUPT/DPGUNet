% 我们的DGG模块，即使用SuperpixelHierarchy生成多层级超像素
% 在此，我们仍以Munich 数据集为例进行演示
% 首先，要对图片进行裁剪，即使用crop_dataset.py裁剪Munich_s1.tif；
% 其次，对裁剪之后的patches进行超像素分割；
% 然后，按需设置相应参数，如超像素数列表（列表长度是层级数），要分割的图像路径，保存路径等；
% 最后，执行该脚本即可！
clear;close all;
% 多层级超像素节点数，即第一个分割图中超像素个数为4096，第二个1024，第三个256，第四个64
n_superpixels=[4096,1024,256,64];
% 要分割的图像文件夹
dir='F:\Dataset\multi_sensor_landcover_classification\crop\';
% 图像名前缀，如图片是Munich_s1_s001.png
img_prefix='Munich_s1_s';
extension = '.png';
save_dir='F:\Dataset\multi_sensor_landcover_classification\segments\';
% 要分割的图像总数
total_num = 624;

for i=1:total_num
    
    num = num2str(i,'%03d');
    % 拼接后的图像名格式类似于: Munich_s1_s001.png
    path = [dir,img_prefix,num,extension];

    I = imread(path);
%     I=load(path);
    I=double(I);
    [h, w] = size(I);
    I = reshape(I, 1, h*w);
    % 归一化
    I=((mapminmax(I)+1)/2)*255;
%     I=mapminmax(I)*255;
    I=reshape(uint8(I),[h,w,1]);

    I = repmat(I, [1,1,3]); % 单通道变三通道
    % gaussian滤波: 用于图像模糊化（去除细节和噪声）
    I = imfilter(I, fspecial('gaussian',[5,5]), 'replicate');

    E=uint8(zeros([h,w]));

    % fine detail structure
    sh=SuperpixelHierarchyMex(I,E,0.0,0.1);
    segmentmaps=zeros(size(n_superpixels,2),h,w);
    for j=1:size(n_superpixels,2)
        GetSuperpixels(sh,n_superpixels(:,j));
        segmentmaps(j,:,:)=sh.label;
    end
    save_path = [save_dir,'segments_s',num,'.mat'];
    save( save_path, 'segmentmaps' );
end





