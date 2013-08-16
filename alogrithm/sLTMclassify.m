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
% label link
label_link = gnd*gnd';
zero_index = (label_link==0);
tmp = link(trainIdx, trainIdx);
tmp(zero_index(trainIdx, trainIdx)) = 0;
link(trainIdx, trainIdx) = tmp;
% link(trainIdx, trainIdx) = label_link(trainIdx, trainIdx);
% LTM
LTMoptions = [];
LTMoptions.maxIter = 100;
LTMoptions.alpha = 50;
LTMoptions.Verbosity = 0;
K = 20;
[fea] = LTM(fea', K, link, LTMoptions);
fea = fea';
% KNN
predict = knnclassify(fea(testIdx,:), fea(trainIdx,:), label(trainIdx), 10, 'cosine');
correct = (predict == label(testIdx));
acc = mean(correct);
disp(acc);