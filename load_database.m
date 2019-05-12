function out=load_database();
% We load the database the first time we run the program.
    v=zeros(10304,400);
   for i=1:40
        a=num2str(i);
        cd(strcat('s',a));
        for j=1:10
            a=imread(strcat(num2str(j),'.pgm'));
           
            v(:,(i-1)*10+j)=reshape(a,size(a,1)*size(a,2),1);
       
        end
        cd ..
    w=uint8(v); % Convert to unsigned 8 bit numbers to save memory. 
end
out=w;