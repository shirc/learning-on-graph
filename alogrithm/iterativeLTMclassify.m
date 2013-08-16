clc;
clear;
% load data
data = load('cora');
fea = data.fea;
gnd = data.gnd;
[~,label] = max(gnd, [], 2);
link = data.link;
% holdout cv
holdout = load('cora_holdout');
trainIdx = holdout.trainIdx;
testIdx = holdout.testIdx;
% LTM options
LTMoptions = [];
LTMoptions.maxIter = 100;
LTMoptions.alpha = 50;
LTMoptions.Verbosity = 0;
K = 20;
pred_gnd = gnd;
for i = 1:10
    % LTM
    if i == 1
        [Pz_d, Pw_z] = LTM(fea', K, link, LTMoptions);
    else
        % update link
        label_link = pred_gnd*pred_gnd';
        zero_index = (label_link==0);
        newlink = link;
        newlink(zero_index) = 0;
        [Pz_d, Pw_z] = LTM(fea', K, newlink, LTMoptions, Pz_d, Pw_z);
    end
    newfea = Pz_d';
    % KNN
    predict = knnclassify(newfea(testIdx,:), newfea(trainIdx,:), label(trainIdx), 10, 'cosine');
    correct = (predict==label(testIdx));
    acc = sum(correct) / size(correct, 1);
    disp(acc);
    % update pred_gnd
    tmp = zeros(sum(testIdx), size(gnd,2));
    ind = sub2ind(size(tmp), 1:sum(testIdx), predict');
    tmp(ind) = 1;
    pred_gnd(testIdx,:) = tmp;
end