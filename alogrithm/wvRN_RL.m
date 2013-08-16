function [ pred ] = wvRN_RL( fea, link, gnd, trainIdx, testIdx )
    [num_inst, num_label] = size(gnd);
    num_train = sum(trainIdx);
    %% parameters
    gamma = 0.99;
    itrnum = 100;
    %% wvRN_RL
    gnd(testIdx, :) = 0;
    gnd(testIdx,:) = ones(num_inst-num_train, 1) * (sum(gnd) / num_train);
    for i = 1:itrnum
        tmp = link*gnd;
        tmp = tmp ./ (sum(tmp,2)*ones(1,num_label));
        gnd(testIdx,:) = (1-gamma)*gnd(testIdx,:) + gamma*tmp(testIdx,:);
    end
    %% assign labels
    [C,I] = max(gnd, [], 2);
    ind = sub2ind([num_inst,num_label], 1:num_inst, I');
    gnd(:,:) = 0;
    gnd(ind) = 1;
    pred = gnd(testIdx,:);
end