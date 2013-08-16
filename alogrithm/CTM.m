function [ pred ] = CTM(fea, link, gnd, trainIdx, testIdx)
    %% parameters
    alpha = 10;
    beta = 10;
    %% initilize Pw_z, Pz_d
    num_inst = size(fea, 1);
    num_label = size(gnd, 2);
    Y_train = gnd(trainIdx,:);
    [~, label_train] = max(Y_train, [], 2);
    nb = NaiveBayes.fit(fea(trainIdx,:), label_train, 'distribution', 'mn');
    Pw_z = cell2mat(nb.Params);
    Pw_z = Pw_z';
    avg = sum(Y_train);
    avg = avg / sum(avg);
    Pz_d = ones(num_inst, 1) * avg;
    Pz_d(trainIdx,:) = Y_train;
    Pz_d = Pz_d';
    %% options
    options = [];
    options.maxIter = 100;
    options.alpha = alpha;
    options.Verbosity = 1;
    options.minIter = 5;
    %% LTM
%     [Pz_d] = LapPLSI2(fea', num_label, link, options, Pz_d, Pw_z, trainIdx, Y_train');
%     [Pz_d] = LTM2(fea', num_label, link, options, Pz_d, Pw_z, trainIdx, Y_train');
    options.beta = beta;
    [Pz_d] = LTM3(fea', num_label, link, options, Pz_d, Pw_z, trainIdx, Y_train');
    %% assign labels
    [~, pred_label] = max(Pz_d, [], 1);
    pred = zeros(sum(testIdx), num_label);
    ind = sub2ind(size(pred), 1:sum(testIdx), pred_label(testIdx));
    pred(ind) = 1;
end

