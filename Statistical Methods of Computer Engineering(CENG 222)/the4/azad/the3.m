N = 38416;
P = 0.259;
Res = 0;
i = 0;
while i < N
    Pois = poissrnd(20);
    R = binornd(Pois,P);
    if R > 6
        Res = Res + 1;
    end
    i = i+1;
end
Out1 = Res ./ N;

N2 = 38416;
a = 0;
b = 13;
c = 0.36; %1/e
Res2 = 0;
j = 0;
while j < N
    Poiss = poissrnd(20);
    l = 0;
    while l < Poiss
        func = 0;
        func2 = c;
        while(func2 > func*exp((-1)*func))
            r = rand;
            r2 = rand;
            func = a + (b-a)*r;
            func2 = c*r2;
        end
        Res2 = Res2 + func;
        l = l +1;
    end
    j = j+1;
end
Out2 = Res2 ./ N;

N3 = 38416;
k = 0;
Res3 = 0;
while k < N
    exp = exprnd(0.5);
    nor = normrnd(0,1);
    Res3 = Res3 + (exp + 2 * nor) ./ (1 + abs(nor));
    k = k+1;
end
Out3 = Res3 ./ N;