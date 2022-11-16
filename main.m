clc;
clear all;
close all;
%% ��ȡͼ��
root='./data';
out_Files = dir(root);%չ��
tempind=0;
img=cell(0);
n=length(out_Files);
%========��ȡ�ļ�========%
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
%% ��ȡ����
% ����:�ڵװ��ֵĶ�ֵͼ�������25ά����������
% ======��ȡ������ת��5*5������ʸ��,��ͼ����ÿ10*10�ĵ���л�����ӣ�������ӳ�һ����=====%
%======��ͳ��ÿ��С������ͼ��������ռ�ٷֱ���Ϊ��������====%
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
%% �����ǩ
class=10;
numberpclass=500;
ann_label=zeros(class,numberpclass*class);
ann_data=img_feature;
for i=1:class
    for j=numberpclass*(i-1)+1:numberpclass*i
        ann_label(i,j)=1;
    end
end

%% ѡ��ѵ�����Ͳ��Լ�
k=rand(1,numberpclass*class);  %����1��5000�е�һ���������
[m,n]=sort(k);         %m���Ѿ�������k��n����������Ҳ����k(n)=m
ntraindata=4500;
ntestdata=500;
train_data=ann_data(:,n(1:ntraindata));
test_data=ann_data(:,n(ntraindata+1:numberpclass*class));
train_label=ann_label(:,n(1:ntraindata));
test_label=ann_label(:,n(ntraindata+1:numberpclass*class));
%% BP�����紴����ѵ���Ͳ���
layer=30;%���������
net=newff(train_data,train_label,layer);%��������
net.trainParam.lr=0.2;%ѧϰ��
net.trainFcn='trainrp';%ѵ������
net.trainParam.epochs=1500;%��������
net.trainParam.goal=0.001;%������
% ����ѵ��
net=train(net,train_data,train_label);
an=sim(net,test_data);
for i=1:length(test_data)
    out(i)=find(an(:,i)==max(an(:,i)));
end
predict_label=out;
%% ��ȷ�ʼ���
[u,v]=find(test_label==1);
label=u';
error=label-predict_label;
accuracy=size(find(error==0),2)/size(label,2);

comparable(1,:)=u(1:20)'-1;
