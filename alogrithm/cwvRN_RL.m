function [ pred ] = cwvRN_RL( fea, link, gnd, trainIdx, testIdx )
    [num_inst, num_label] = size(gnd);
    num_train = sum(trainIdx);
    %% parameters
    gamma = 0.99;
    itrnum = 100;
    %% wvRN_RL
    gnd(testIdx, :) = 0;
    gnd(testIdx,:) = ones(num_inst-num_train, 1) * (sum(gnd) / num_train);
    S = eye(num_label);
    D = diag(sum(link,2));
    for k = 1:10
        for i = 1:itrnum
            tmp = link*gnd*S;
            tmp = tmp ./ (sum(tmp,2)*ones(1,num_label));
            gnd(testIdx,:) = (1-gamma)*gnd(testIdx,:) + gamma*tmp(testIdx,:);
        end
        for i = 1:itrnum
            S = S + gamma*gnd'*(D*gnd*S - link*gnd)./(sum(link*(gnd.^2))'*ones(1,num_label));
        end
    end
    %% assign labels
    [C,I] = max(gnd, [], 2);
    ind = sub2ind([num_inst,num_label], 1:num_inst, I');
    gnd(:,:) = 0;
    gnd(ind) = 1;
    pred = gnd(testIdx,:);
end