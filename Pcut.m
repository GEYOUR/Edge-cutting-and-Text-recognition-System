function  output_img = Pcut(path, shape)
close all;

%shape = input("shape:");
%path = "E:\Collected\[Courses]\(3-1)�����Ӿ�\Teamwork\Candidate_imgs\����\0129145.jpg"
reply =  str2num(shape);

%���ڻ���任Բ�ζ�λ��ͼƬ�и�
       if (reply==1)
       
           I = imread(path);                          
           I = rgb2gray(I);
           [m,n] = size(I);                                % ��ȡͼƬ�ߴ�
          
           %figure(1),imshow(I);                            % ��ʾԭͼ
       
           minr = round(m/8);                              % ����Բ�γߴ緶Χ������С�뾶�����뾶     
           maxr = round(m/1.8);
           [c,r] = imfindcircles(I,[minr,maxr],'ObjectPolarity','dark','Sensitivity',0.99);  %��imfindcircles��������Բ��ͼ��������ԵΪ��ɫ��������Ϊ0.99
        
           rad = max(r);                                   % ��������Բ������ȡ���뾶
           num = find(r==rad);                             % ȷ�����뾶�ڰ뾶�����е����
           cent = [c(num,1),c(num,2)];                     % �뾶�������Բ���������е����һ�£��Ӷ���ȡԲ������
           figure(2)
           imshow(I);title('��λ')                         % ��ʾ��λͼ��
           viscircles(cent, rad,'EdgeColor','g');          % ����ɫ�����ڶ�λͼ���ϻ�Ȧ��ʾ
         
           figure(3)
           pic = imcrop(I,[cent(1)-rad-50 cent(2)-rad-70 2*rad+200 2*rad+180]);   % ����ȷ���İ뾶��Բ�ģ������и������һ����ƫ����
           imshow(pic);title('Ŀ����ȡ');                                          %  ��ʾ�и���
      
       end
       
       
       
 %���ڻ���任ֱ�߼�ⶨλ���Ƽ�ʵ��ͼƬ�и�      
       if(reply==2)

             I = imread(path);                                      
             I = rgb2gray(I);
             figure(1)
             [m,n] = size(I);
             imshow(I);title('ԭͼ')                       % ��ʾԭͼ��
             
             I=imcrop(I,[m/6,n/6,5*n/6,2*n/3]);            % ���и����С��ⷶΧ���ų����ӵ�ֱ
       
             BW = edge(I,'canny');                         % canny��Ե���õ���ֵͼ��
             
             [m,n] = size(BW);                             % ��ȡ���и��ͼ���
       
             [H,T,R] = hough(BW);                          % ��ͼ����л���任
             figure(2)
             imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');title('����ռ�');       % ��ʾ����ռ����ߴ�
             xlabel('\theta'), ylabel('\rho');             % �������������
             axis on, axis normal, hold on;
             
             cei = ceil(0.6*max(H(:)));                    % ����任��ֵ������ֵ����
             P = houghpeaks(H,5,'threshold',cei);          % ��ȡ��ֵ������Ϊ5
             x = T(P(:,2));                                % ��ȡ��ֵ��plotͼ���еĺ�����x
             y = R(P(:,1));                                % ��ȡ��ֵ��plotͼ���еĺ�����y
       
             plot(x,y,'s','color','white');                % �԰�ɫ�����ֵ����
             colormap(hot);                                % ��������������ʽչʾ
       
% Find lines and plot them
            lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',10);              % ��houghlines������ȡֱ����Ϣ���ϲ�ֱ�ߵ���С�����5�����ֱ�߳��ȵ�������10
            figure(3), imshow(I),title('�ֱ�߶�λ'); hold on                    % ��ʾֱ�߻���ͼ
            max_len = 0;                                                          %  �ֱ�߱�����ֵ��0  

            for k = 1:length(lines)                                               % ���λ��Ƽ�������ֱ��
                xy = [lines(k).point1; lines(k).point2];                          % ȷ����ֱ������
                plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','g');                  % ����ֱ�߿����2����ɫ��Ϊ��ɫ
 
% Plot beginnings and ends of lines
                plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');         % ���Ƹ�ֱ����ֹ���꣬��ʼ��Ϊ��ɫ���յ�Ϊ��ɫ���ߴ�Ϊ2
                plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');            
 
% Determine the endpoints of the longest line segment 
                len = norm(lines(k).point1 - lines(k).point2);g(1)=xy(4);         % �����ֱ�ߵĳ���
         
                if ( len > max_len)                                               % ������ѭ��ֱ�ߴ�����һ��ֱ�ߵĳ���
                                                                                  
                    max_len = len;                                                % �򽫱��ε��߳��������ֱ�߱���
                    xy_long = xy;                                                 % ������ֱ�ߵ����긳������߳�
    
                end
            end
                plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');     % �����ֱ�ߣ���ɫΪcyan

                if (m/3>xy_long(3))                                               % ���´��룬��ͼƬ�߶ȷֳ����ȷ��и�����ֱ�����ڵ�����,ȡ��ͬ��ֵ��ͼƬ�����и�
    
                    pic=imcrop(I,[n/14,xy_long(3)+m/(20),4*max_len,m/(2.5)]);   % �и��������Ϊ��x,y��=[n/14,xy_long(3)+m/(7.5)],��3.5*max_len����m/(3.2)
                    figure(4)                                                    
                    imshow(pic);title('Ŀ����ȡ');
              
                end
                
                if(m/3<xy_long(3) && 2*m/3>xy_long(3))
        
                    pic=imcrop(I,[n/6,m/6,2*n/3,2*m/3]);                           % �и��������Ϊ��x,y��=[n/6,m/6],��2*n/3����2*m/3.���˴������ж��ֱ��������ͼƬ���м�λ�ø��������ֱ�������Ե��и�ͼƬ���м�λ�ã�
                    figure(4)
                    imshow(pic);title('Ŀ����ȡ');
           
                end
                
                if(2*m/3<xy_long(3))
       
                    pic=imcrop(I,[n/14,xy_long(3)-m/(2.2),3.5*max_len,m/3]);          % �и��������Ϊ��x,y��=[n/14,xy_long(3)+m/(7.5)],��3.5*max_len����m/(3.2)
                    figure(4)
                    imshow(pic);title('Ŀ����ȡ');
                end
             
       end
       mid_img = pic;
       BW = edge(pic,'sobel');                         % canny��Ե���õ���ֵͼ��
       figure(5),imshow(BW);title("output image");
       output_img = BW;