function  output_img = Pcut(path, shape)
close all;

%shape = input("shape:");
%path = "E:\Collected\[Courses]\(3-1)机器视觉\Teamwork\Candidate_imgs\车牌\0129145.jpg"
reply =  str2num(shape);

%基于霍夫变换圆形定位及图片切割
       if (reply==1)
       
           I = imread(path);                          
           I = rgb2gray(I);
           [m,n] = size(I);                                % 获取图片尺寸
          
           %figure(1),imshow(I);                            % 显示原图
       
           minr = round(m/8);                              % 搜索圆形尺寸范围，即最小半径和最大半径     
           maxr = round(m/1.8);
           [c,r] = imfindcircles(I,[minr,maxr],'ObjectPolarity','dark','Sensitivity',0.99);  %用imfindcircles函数搜索圆形图案，检测边缘为黑色，灵敏度为0.99
        
           rad = max(r);                                   % 已搜索的圆形中提取最大半径
           num = find(r==rad);                             % 确定最大半径在半径数组中的序号
           cent = [c(num,1),c(num,2)];                     % 半径的序号与圆心在数组中的序号一致，从而获取圆心坐标
           figure(2)
           imshow(I);title('定位')                         % 显示定位图像
           viscircles(cent, rad,'EdgeColor','g');          % 以绿色绘制在定位图像上画圈显示
         
           figure(3)
           pic = imcrop(I,[cent(1)-rad-50 cent(2)-rad-70 2*rad+200 2*rad+180]);   % 根据确定的半径、圆心，进行切割，并配有一定的偏移量
           imshow(pic);title('目标提取');                                          %  显示切割结果
      
       end
       
       
       
 %基于霍夫变换直线检测定位车牌及实现图片切割      
       if(reply==2)

             I = imread(path);                                      
             I = rgb2gray(I);
             figure(1)
             [m,n] = size(I);
             imshow(I);title('原图')                       % 显示原图像
             
             I=imcrop(I,[m/6,n/6,5*n/6,2*n/3]);            % 粗切割，以缩小检测范围和排除夹杂的直
       
             BW = edge(I,'canny');                         % canny边缘检测得到二值图像
             
             [m,n] = size(BW);                             % 获取粗切割后图像尺
       
             [H,T,R] = hough(BW);                          % 对图像进行霍夫变换
             figure(2)
             imshow(H,[],'XData',T,'YData',R,'InitialMagnification','fit');title('霍夫空间');       % 显示霍夫空间曲线簇
             xlabel('\theta'), ylabel('\rho');             % 横纵坐标的命名
             axis on, axis normal, hold on;
             
             cei = ceil(0.6*max(H(:)));                    % 霍夫变换峰值检测的阈值设置
             P = houghpeaks(H,5,'threshold',cei);          % 获取峰值，数量为5
             x = T(P(:,2));                                % 获取峰值在plot图像中的横坐标x
             y = R(P(:,1));                                % 获取峰值在plot图像中的横坐标y
       
             plot(x,y,'s','color','white');                % 以白色框出峰值坐标
             colormap(hot);                                % 曲线以热量的形式展示
       
% Find lines and plot them
            lines = houghlines(BW,T,R,P,'FillGap',5,'MinLength',10);              % 以houghlines函数获取直线信息，合并直线的最小间距设5，检测直线长度的下限设10
            figure(3), imshow(I),title('最长直线定位'); hold on                    % 显示直线绘制图
            max_len = 0;                                                          %  最长直线变量初值设0  

            for k = 1:length(lines)                                               % 依次绘制检测的所有直线
                xy = [lines(k).point1; lines(k).point2];                          % 确定各直线坐标
                plot(xy(:,1),xy(:,2),'LineWidth',2,'Color','g');                  % 绘制直线宽度设2，颜色设为绿色
 
% Plot beginnings and ends of lines
                plot(xy(1,1),xy(1,2),'x','LineWidth',2,'Color','yellow');         % 绘制各直线起止坐标，起始点为黄色，终点为红色，线粗为2
                plot(xy(2,1),xy(2,2),'x','LineWidth',2,'Color','red');            
 
% Determine the endpoints of the longest line segment 
                len = norm(lines(k).point1 - lines(k).point2);g(1)=xy(4);         % 计算各直线的长度
         
                if ( len > max_len)                                               % 若本次循环直线大于上一次直线的长度
                                                                                  
                    max_len = len;                                                % 则将本次的线长赋给最大直线变量
                    xy_long = xy;                                                 % 将本次直线的坐标赋给最大线长
    
                end
            end
                plot(xy_long(:,1),xy_long(:,2),'LineWidth',2,'Color','cyan');     % 绘制最长直线，颜色为cyan

                if (m/3>xy_long(3))                                               % 以下代码，将图片高度分成三等分切割，根据最长直线落在的区域,取不同的值对图片进行切割
    
                    pic=imcrop(I,[n/14,xy_long(3)+m/(20),4*max_len,m/(2.5)]);   % 切割起点坐标为（x,y）=[n/14,xy_long(3)+m/(7.5)],宽3.5*max_len，高m/(3.2)
                    figure(4)                                                    
                    imshow(pic);title('目标提取');
              
                end
                
                if(m/3<xy_long(3) && 2*m/3>xy_long(3))
        
                    pic=imcrop(I,[n/6,m/6,2*n/3,2*m/3]);                           % 切割起点坐标为（x,y）=[n/6,m/6],宽2*n/3，高2*m/3.（此处可以判断最长直线在整张图片的中间位置附近，因此直接主观性的切割图片的中间位置）
                    figure(4)
                    imshow(pic);title('目标提取');
           
                end
                
                if(2*m/3<xy_long(3))
       
                    pic=imcrop(I,[n/14,xy_long(3)-m/(2.2),3.5*max_len,m/3]);          % 切割起点坐标为（x,y）=[n/14,xy_long(3)+m/(7.5)],宽3.5*max_len，高m/(3.2)
                    figure(4)
                    imshow(pic);title('目标提取');
                end
             
       end
       mid_img = pic;
       BW = edge(pic,'sobel');                         % canny边缘检测得到二值图像
       figure(5),imshow(BW);title("output image");
       output_img = BW;