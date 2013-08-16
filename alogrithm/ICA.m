function [ pred ] = ICA( fea, link, gnd, trainIdx, testIdx )
    %% parameters
    iter_num = 10;
    num_label = size(gnd, 2);
    num_test = sum(testIdx);
    %% learn
    [C, label] = max(gnd, [] , 2);
    gnd(testIdx,:) = 0;
    rel_fea = link * gnd;
    model = train(label(trainIdx,:), [fea(trainIdx, :) rel_fea(trainIdx, :)], '-c 1 -q');
    %% inference
    pred = zeros(num_test, num_label);
    for i = 1:iter_num
        [predict_label] = predict(label(testIdx,:), [fea(testIdx, :) rel_fea(testIdx, :)], model);
        ind = sub2ind([num_test,num_label], 1:num_test, predict_label');
        pred(:,:) = 0;
        pred(ind) = 1;
        gnd(testIdx, :) = pred;
        rel_fea = link*gnd;
        % re-train
        model = train(label(trainIdx,:), [fea(trainIdx, :) rel_fea(trainIdx, :)], '-c 1 -q');
    end
end