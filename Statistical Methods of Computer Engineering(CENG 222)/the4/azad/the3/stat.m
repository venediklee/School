%QUESTION 1 %

jointFunction= @(w,s) w.*s.*exp(-w-s);
error = 0.005; %given in the question
zVal = norminv((1-0.95)/2);


pMinion = 0.259259; % found by integration
lambda = 5*4; % since our time is 5 hours


numberOfSuccess = 0;% result
N = ceil((0.25)* (power(zVal/error,2))); %Find how many tries is needed

for i=1:N;
    minionCount = 0;
    poissonRand = poissrnd(lambda);
    for j=1:poissonRand;% for each minion 
        U = rand()<pMinion; % Random Variable for Bernoulli distribution
        minionCount = minionCount +U;
    end;
    
    if minionCount>6% check if the condition is satisfied
        numberOfSuccess = numberOfSuccess+1;
    end;
  
end;

expectedSucc = numberOfSuccess/N;
disp(expectedSucc); % QUESTION 1 answer

%QUESTION 2 %

f = @(x) exp(-x)*x;
totalWeight = 0;
limit_a = 0; %lower_limit
limit_b = 10; %upper_limit, NOTE f(10)<0.001
limit_c = 1/exp(1); % c = max value of f

%By using rejection method. 
for i=1:N;
    weightPerUnit = 0;
    poissonRand = poissrnd(lambda);
    for j=1:poissonRand;
        X=0;
        Y=1/exp(1);
        while Y > f(X) % Try till (X,Y) is in ACCEPT region.
            U= rand();
            V= rand();
            X = limit_a + (limit_b-limit_a)*U;
            Y = limit_c*V;
        end;
        % Now a valid (X,Y) pair is found.
        weightPerUnit = weightPerUnit+ X;
    end;
    totalWeight=totalWeight+ weightPerUnit;
    
end;
disp(totalWeight/N); %QUESTION 2 - Estimated total weight

%QUESTION 3 %
totalValue = 0;
for i=1:N;
    A = exprnd(2);
    B= normrnd(0,1);
    totalValue=totalValue+ ((A+2*B)/(1+abs(B)));
end;
disp(totalValue/N); %Question 3 answer,expected.