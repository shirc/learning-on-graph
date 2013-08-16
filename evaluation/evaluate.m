function [ acc ] = evaluate( dataset, alg )
    data = load(['dataset/' dataset]);
    cv = load(['dataset/' dataset '_holdout']);
    fea = data.fea;
    link = data.link;
    gnd = data.gnd;
    trainIdx = cv.trainIdx;
    testIdx = cv.testIdx;
    pred = alg(fea, link, gnd, trainIdx, testIdx);
    acc = 1 - sum(sum(abs(gnd(testIdx,:) - pred))) / 2 / sum(testIdx);
end