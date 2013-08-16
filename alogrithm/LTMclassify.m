function [ pred ] = LTMclassify(fea, link, gnd, trainIdx, testIdx)
    num_test = sum(testIdx);
    num_label = size(gnd, 2);
    [~, label] = max(gnd, [], 2);
    % LTM
    LTMoptions = [];
    LTMoptions.maxIter = 100;
    LTMoptions.alpha = 50;
    LTMoptions.Verbosity = 0;
    K = 20;
    [new_fea] = LTM(fea', K, link, LTMoptions);
    new_fea = sparse(new_fea');
    % linear
    pred = zeros(num_test, num_label);
    model = train(label(trainIdx,:), new_fea(trainIdx, :), '-c 1 -q');
    [predict_label] = predict(label(testIdx,:), new_fea(testIdx, :), model);
    ind = sub2ind([num_test, num_label], 1:num_test, predict_label');
    pred(:,:) = 0;
    pred(ind) = 1;
end