%Q1%

lambda = 7*5; %% 7 hrs

f= @(w,s) w.*s.*exp(-w-s);
maxY = @(w) w./2;
prob = integral2(f, 0, inf, 0, maxY); % .2593

count = ceil(0.25 * power(1.96/0.005, 2));%%crit=1.96,error=0.005

numSuccess = 0;
poisR = poissrnd(lambda);
for i=1:count
    numCaught = 0;
    numCaught = binornd(poisR, prob);
    if numCaught > 8
        numSuccess = numSuccess+1;
    end
end
disp(numSuccess/count); 


% Q2 %

totalWeight = 0;
ceiling = 12; 
floor = 0;
multiplier = exp(-1);

lambda10 = 10*5;% 10hrs
for i=1:count
	poisR10 = poissrnd(lambda10);
    for j=1:poisR10
        X = 0;
        Y = multiplier;
        while Y > X*exp(-X)
            X = floor + (ceiling-floor)*rand;
            Y = multiplier*rand;
        end
        totalWeight=totalWeight + X;
    end
end
disp(totalWeight/count);  % around 99.6897


% Q3 %
total = 0;
for i=1:count
    A = exprnd(1/2);
    B = normrnd(0,1);
    total = total + (2*A+3*B)/(3+2*abs(B));
end
disp(total/count);% around 0.23380