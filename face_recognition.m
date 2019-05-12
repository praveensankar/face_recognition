%% Face recognition
% This algorithm uses the eigenface system (based on pricipal component
% analysis - PCA) to recognize faces.
clear
cd '/media/praveen/praveen/project/mini project/image_set'

w=load_database();

%enter the input image path
x=input('Enter the path  : ');

%% Initializations
a=imread(x);
vv=reshape(a,size(a,1)*size(a,2),1);
r=uint8(vv);                          % r contains the image we later on will use to test the algorithm
v=w(:,:);         
N=40;                      % Number of signatures used for each image.
%% Subtracting the mean from v
O=uint8(ones(1,size(v,2)));
m=uint8(mean(v,2));                 % m is the maen of all images.
vzm=v-uint8(single(m)*single(O));
vz1=vzm;
% vzm is v with the mean removed.
%title('Mean removed images','FontWeight','bold','Fontsize',18,'color','green');
%imtool(reshape(vz1,size(vz1)));
%% Calculating eignevectors of the correlation matrix
% We are picking N of the 400 eigenfaces.
L=single(vzm)'*single(vzm);
[V,D]=eig(L);
V=single(vzm)*V;
V=V(:,end:-1:end-(N-1));            % Pick the eignevectors corresponding to the 10 largest eigenvalues. 
%imtool(V);

%% Calculating the signature for each image
cv=zeros(size(v,2),N);
for i=1:size(v,2);
    cv(i,:)=single(vzm(:,i))'*V 
end


%% Recognition 
%  Now, we run the algorithm and see if we can correctly recognize the face. 
subplot(121); 
imshow(reshape(r,112,92));title('Looking for ...','FontWeight','bold','Fontsize',16,'color','red');

subplot(122);
p=r-m;                              % Subtract the mean
s=single(p)'*V
z=[];
for i=1:size(v,2)
    z=[z,norm(cv(i,:)-s,2)];
    if(rem(i,20)==0),imshow(reshape(v(:,i),112,92)),end;
    drawnow;
end

[a,i]=min(z);
subplot(122);
imshow(reshape(v(:,i),112,92));title('Found!','FontWeight','bold','Fontsize',16,'color','blue');


