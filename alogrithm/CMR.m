function [ pred ] = CMR( fea, link, gnd, trainIdx, testIdx )
    [num_inst, num_label] = size(gnd);
    num_train = sum(trainIdx);
    %% parameters
    gamma = 0.99;
    itrnum = 10;
    lambda = 10;
    %% CMR
    gnd(testIdx, :) = 0;
    gnd(testIdx,:) = ones(num_inst-num_train, 1) * (sum(gnd) / num_train);
    ap = zeros(num_inst, num_label);
    for i = 1:itrnum
        for j = 1:100
            tmp = link*gnd;
            tmp = ap + lambda*tmp;
            tmp = tmp ./ (sum(tmp,2)*ones(1,num_label));
            gnd(testIdx,:) = (1-gamma)*gnd(testIdx,:) + gamma*tmp(testIdx,:);
        end
        [C,class] = max(gnd, [], 2);
        nb = NaiveBayes.fit(fea, class, 'distribution', 'mn');
        ap = posterior(nb, fea);
    end
    %% assign labels
    [C,I] = max(gnd, [], 2);
    ind = sub2ind([num_inst,num_label], 1:num_inst, I');
    gnd(:,:) = 0;
    gnd(ind) = 1;
    pred = gnd(testIdx,:);
end