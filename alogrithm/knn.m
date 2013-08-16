clc;
clear;
% load data
data = load('citeseer');
fea = data.fea;
gnd = data.gnd;
[~,gnd] = max(gnd, [], 2);
link = data.link;
% holdout cv
holdout = load('citeseer_holdout');
trainIdx = holdout.trainIdx;
testIdx = holdout.testIdx;
% Bayesian
nb = NaiveBayes.fit(fea(trainIdx,:), gnd(trainIdx), 'distribution', 'mn');
predict = nb.predict(fea(testIdx,:));
correct = (predict == gnd(testIdx));
acc = mean(correct);
disp(acc);
% wvRN_RL
predict = wvRN_RL(link, data.gnd, trainIdx, testIdx);
[~,predict] = max(predict, [], 2);
correct = (predict == gnd(testIdx));
acc = mean(correct);
disp(acc);
% KNN
predict = knnclassify(fea(testIdx,:), fea(trainIdx,:), gnd(trainIdx), 10, 'cosine');
correct = (predict == gnd(testIdx));
acc = mean(correct);

