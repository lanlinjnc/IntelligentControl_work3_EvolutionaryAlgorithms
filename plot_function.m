x=-1:0.001:1;
y=-1:0.001:1;
[X,Y]=meshgrid(x,y);%生成网格，构造X,Y矩阵
Z=X.^2+Y.^2-0.3*cos(X*3*3.1415)-0.4*cos(Y*4*3.1415)+0.7;%f(X,Y)
mesh(X,Y,Z);%以网格状绘制图像
title('{$f(x,y)=x^{2}+y^{2}-0.3\cos(3\pi x)-0.4\cos(4\pi y)+0.7$}','interpreter','latex');
%以latex文档的形式载入title
