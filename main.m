clc;
clear all;
close all;
%% 读取图像
root='./data';
out_Files = dir(root);%展开
tempind=0;
img=cell(0);
n=length(out_Files);
%========读取文件========%
for i = 1:n
    if strcmp(out_Files(i).name,'.')|| strcmp(out_Files(i).name,'..')
    else
        rootpath=strcat(root,'/',out_Files(i).name);
        in_filelist=dir(rootpath);
        ni=length(in_filelist);
        for j=1:ni
            if strcmp(in_filelist(j).name,'.')|| strcmp(in_filelist(j).name,'..')|| strcmp(in_filelist(j).name,'Desktop_1.ini')|| strcmp(in_filelist(j).name,'Desktop_2.ini')
            else
                tempind=tempind+1;
                img{tempind}=imread(strcat(rootpath,'/',in_filelist(j).name));
            end
        end
    end
end
%% 提取特征
% 输入:黑底白字的二值图像。输出：25维的网格特征
% ======提取特征，转成5*5的特征矢量,把图像中每10*10的点进行划分相加，进行相加成一个点=====%
%======即统计每个小区域中图像象素所占百分比作为特征数据====%
 for i=1:length(img)
    bw2=im2bw(img{i},graythresh(img{i}));
    bw_7050=imresize(bw2,[50,50]); 
        for cnt=1:5
            for cnt2=1:5
                Atemp=sum(bw_7050(((cnt*10-9):(cnt*10)),((cnt2*10-9):(cnt2*10))));        
                lett((cnt-1)*5+cnt2)=sum(Atemp);                                     
            end
        end
    lett=((100-lett)/100);
    lett=lett'; 
    img_feature(:,i)=lett; 
end
%% 构造标签
class=10;
numberpclass=500;
ann_label=zeros(class,numberpclass*class);
ann_data=img_feature;
for i=1:class
    for j=numberpclass*(i-1)+1:numberpclass*i
        ann_label(i,j)=1;
    end
end

%% 选定训练集和测试集
k=rand(1,numberpclass*class);  %生成1行5000列的一个随机矩阵
[m,n]=sort(k);         %m是已经排序后的k，n是其索引，也就是k(n)=m
ntraindata=4500;
ntestdata=500;
train_data=ann_data(:,n(1:ntraindata));
test_data=ann_data(:,n(ntraindata+1:numberpclass*class));
train_label=ann_label(:,n(1:ntraindata));
test_label=ann_label(:,n(ntraindata+1:numberpclass*class));
%% BP神经网络创建，训练和测试
layer=30;%隐含层个数
net=newff(train_data,train_label,layer);%创建网络
net.trainParam.lr=0.2;%学习率
net.trainFcn='trainrp';%训练方法
net.trainParam.epochs=1500;%迭代次数
net.trainParam.goal=0.001;%误差参数
% 网络训练
net=train(net,train_data,train_label);
an=sim(net,test_data);
for i=1:length(test_data)
    out(i)=find(an(:,i)==max(an(:,i)));
end
predict_label=out;
%% 正确率计算
[u,v]=find(test_label==1);
label=u';
error=label-predict_label;
accuracy=size(find(error==0),2)/size(label,2);

comparable(1,:)=u(1:20)'-1;
