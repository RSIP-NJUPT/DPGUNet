% ���ǵ�DGGģ�飬��ʹ��SuperpixelHierarchy���ɶ�㼶������
% �ڴˣ���������Munich ���ݼ�Ϊ��������ʾ
% ���ȣ�Ҫ��ͼƬ���вü�����ʹ��crop_dataset.py�ü�Munich_s1.tif��
% ��Σ��Բü�֮���patches���г����طָ
% Ȼ�󣬰���������Ӧ�������糬�������б��б����ǲ㼶������Ҫ�ָ��ͼ��·��������·���ȣ�
% ���ִ�иýű����ɣ�
clear;close all;
% ��㼶�����ؽڵ���������һ���ָ�ͼ�г����ظ���Ϊ4096���ڶ���1024��������256�����ĸ�64
n_superpixels=[4096,1024,256,64];
% Ҫ�ָ��ͼ���ļ���
dir='F:\Dataset\multi_sensor_landcover_classification\crop\';
% ͼ����ǰ׺����ͼƬ��Munich_s1_s001.png
img_prefix='Munich_s1_s';
extension = '.png';
save_dir='F:\Dataset\multi_sensor_landcover_classification\segments\';
% Ҫ�ָ��ͼ������
total_num = 624;

for i=1:total_num
    
    num = num2str(i,'%03d');
    % ƴ�Ӻ��ͼ������ʽ������: Munich_s1_s001.png
    path = [dir,img_prefix,num,extension];

    I = imread(path);
%     I=load(path);
    I=double(I);
    [h, w] = size(I);
    I = reshape(I, 1, h*w);
    % ��һ��
    I=((mapminmax(I)+1)/2)*255;
%     I=mapminmax(I)*255;
    I=reshape(uint8(I),[h,w,1]);

    I = repmat(I, [1,1,3]); % ��ͨ������ͨ��
    % gaussian�˲�: ����ͼ��ģ������ȥ��ϸ�ں�������
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





