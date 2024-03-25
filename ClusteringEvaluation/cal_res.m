function res = cal_res(idx,Y)

res = zeros(3,1);

[newIndx] = bestMap(Y,idx);
nmi = MutualInfo(Y,newIndx);
acc = mean(Y==newIndx);
purity = purFuc(Y, newIndx);

res = [acc, nmi, purity];
end