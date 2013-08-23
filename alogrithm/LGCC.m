function [ Pz_d ] = LGCC( Pw_z, Pz_d, Qz_d, X, W, idx, alpha, beta )
    % OmegaLA   
    [K, N] = size(Qz_d);
    DCol = full(sum(W, 2));
    D = spdiags(DCol, 0, N, N);
    L = D - W;
    L = alpha*L;
    dLen = full(sum(X,1));
    Omega = spdiags(dLen', 0, N, N);
    A = spdiags(idx, 0, N, N);
    OmegaL = Omega + L + beta*A;
    % init
    Pd = sum(X)./sum(X(:));
    Pd = full(Pd);
    Pw_d = mex_Pw_d(X,Pw_z,Pz_d);
    ZERO_OFFSET = 1e-200;
    maxIter = 100;
    % EM
    logL = mex_logL(X,Pw_d,Pd);
    logL = logL - sum(sum((log(Pz_d + ZERO_OFFSET)*L).*Pz_d));
    logPQ = log((Pz_d + ZERO_OFFSET)./(Qz_d + ZERO_OFFSET));
    logL = logL - 0.5*beta*sum(sum((Pz_d - Qz_d).*logPQ));
    pre_Pz_d = Pz_d;
    pre_logL = logL;
    for iter = 1:maxIter
        [Pw_z, Pz_d] = mex_EMstep(X, Pw_d, Pw_z, Pz_d);
        Pz_d = Pz_d.*repmat(dLen, K, 1);
        Pz_d = (OmegaL\(Pz_d'+beta*A*Qz_d'))';
        Pw_d = mex_Pw_d(X, Pw_z, Pz_d);
        logL = mex_logL(X,Pw_d,Pd);
        logL = logL - sum(sum((log(Pz_d + ZERO_OFFSET)*L).*Pz_d));
        logPQ = log((Pz_d + ZERO_OFFSET)./(Qz_d + ZERO_OFFSET));
        logL = logL - 0.5*beta*sum(sum((Pz_d - Qz_d).*logPQ));
        delta = (pre_logL - logL)/pre_logL;
        disp(['iter:', num2str(iter), ' logL:', num2str(logL), ' delta:', num2str(delta)]);
        if delta < 1e-7
            if delta < 0
                Pz_d = pre_Pz_d;
            end
            break;
        else
            pre_Pz_d = Pz_d;
            pre_logL = logL;
        end
    end
end

