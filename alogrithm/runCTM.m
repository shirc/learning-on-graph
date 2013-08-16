clear;
clc;
% load data
data = load('citeseer');
fea = data.fea;
gnd = data.gnd;
link = data.link;
% holdout cv
holdout = load('citeseer_holdout');
trainIdx = holdout.trainIdx;
testIdx = holdout.testIdx;
% run CTM
predict = CTM(fea, gnd(trainIdx,:), link, trainIdx, 10, 1);
[~,gnd] = max(gnd, [], 2);
[~,predict] = max(predict, [], 2);
correct = (predict == gnd(testIdx));
acc = mean(correct);
disp(acc);