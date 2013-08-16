function [ ] = calc_sim( dataset )
    data = load(['dataset/' dataset]);
    fea = data.fea;
    link = data.link;
    gnd = data.gnd;
    num_inst = size(fea, 1);
    tmp = zeros(num_inst, num_inst);
    sigma = 10;
    sim = exp(-sigma*squareform(pdist(fea)));
    [sort_V, sort_I] = sort(sim, 2, 'descend');
    for i = 1:num_inst
        tmp(sort_I(i, 1:10), i) = sort_V(i, 1:10);
        tmp(i, sort_I(i, 1:10)) = sort_V(i, 1:10)';
    end
    link = link + sparse(tmp);
    save(['dataset/' dataset '.mat'], 'fea', 'link', 'gnd');
end