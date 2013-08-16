function [ pred ] = LP( fea, link, gnd, trainIdx, testIdx )
    W = [link(trainIdx,:);link(testIdx,:)];
    W = [W(:,trainIdx) W(:,testIdx)];
    fl = gnd(trainIdx,:);
    pred = harmonic_function(W, fl);
    [C,I] = max(pred, [], 2);
    ind = sub2ind(size(pred), 1:size(pred,1), I');
    pred(:,:) = 0;
    pred(ind) = 1;
end