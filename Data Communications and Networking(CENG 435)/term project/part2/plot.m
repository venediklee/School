x=[5,15,38];
y=[651.345, 2531.647, 8532.620];
e=[7.754, 14.798, 0];

errorbar(x,y,e);
xlabel('Packet Loss(percent)');
ylabel('Total Transfer Time(seconds)');
title('Packet Loss vs Total Transfer Time');
print('experiment1Graph.png')
pause;



x=[5];
y=[551.179];
e=[6.783];

errorbar(x,y,e);
xlabel('Packet Loss(percent)');
ylabel('Total Transfer Time(seconds)');
title('Packet Loss vs Total Transfer Time');
print('experiment2Graph.png')
pause;