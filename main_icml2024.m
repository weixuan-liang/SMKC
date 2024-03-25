clc
clear

prefix = './';
addpath(genpath('./'));

dataNameSet =  {'Cifar10'};

for name = 1
    load(['.\dataset\', dataNameSet{name}, '.mat'])
    numker = length(X);
    num = length(X{1});
    num_landmark = 3 * ceil(sqrt(num));
    rng(1);
    index = sort(datasample(1:num, num_landmark, 'replace', false));
    P = zeros(num, num_landmark,numker);
    W = zeros(num_landmark,num_landmark,numker);
    data_concatenate = [];
    for ker = 1:numker
        data_temp = X{ker};
        data_temp = pre_process(data_temp);
        sample_row = data_temp(index,:);
        P(:,:,ker) = create_kernel(data_temp, sample_row);
    end
    
    fprintf('DataName: %s\n',dataNameSet{name});
    numclass = length(unique(Y));
    Y(Y<1)=numclass;
      
    tic;
    [G_star,obj] = fusion_p_Kernel(P,numclass);
    [u,d,v] = svds(G_star, numclass);
    timecost = toc;
    res_mean = myNMIACC(u, Y, numclass);
    
    
end