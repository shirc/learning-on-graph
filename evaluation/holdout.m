function [] = holdout( dataset, p)
    % load data
    data = load(dataset);
    fea = data.fea;
    gnd = data.gnd;
    [~,group] = max(gnd, [], 2);
    % holdout cv
    num_inst = size(fea,1);
    cv = cvpartition(group, 'holdout', 1-p);
    trainIdx = training(cv, 1);
    testIdx = test(cv, 1);
    save(['dataset/' dataset '_holdout.mat'], 'trainIdx', 'testIdx');
end