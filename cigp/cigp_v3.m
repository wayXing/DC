function model = cigp_v3(xTr, yTr, xTe)
%train cigp model and predict
% data is not normalized
% add a forwad function to cigp_v2

    kernel = 'ard';

    dist = pdist2(xTr,min(xTr));
    [~,index] = sort(dist);
    xTr = xTr(index,:);
    yTr = yTr(index,:);


    a0 = 1e-3; b0 = 1e-3;
    [N,d] = size(xTr);

    m = size(yTr,2);
    D = yTr*yTr';
    assert(size(xTr,1)==size(yTr,1),'inconsistent data');
    log_bta = log(1/var(yTr(:)));
%     log_bta = log(1/eps);
    
    log_l = zeros(d,1);
%     log_l = log(mean(xTr)'/10);
    
    %log_l = 2*log(median(pdist(Xtr)))*ones(d,1);
%     log_sigma = 0;
    log_sigma = log(1e-4);  %initial noise variance
    log_sigma0 = log(1e-12);    %initial noise variance in kernel function
    params = [log_l;log_sigma;log_sigma0;log_bta];
    fastDerivativeCheck(@(params) log_evidence_CIGP(params,a0,b0,m, xTr, D, kernel), params);
    %max_iter = 1000;
    %new_param = minimize(param, @(param) log_evidence_lower_bound(param, x, y, m), max_iter);
    opt = [];
    opt.MaxIter = 300;
    opt.MaxFunEvals = 10000;
    new_params = minFunc(@(params) log_evidence_CIGP(params,a0,b0,m, xTr, D, kernel), params,opt);
    [ker_param,idx] = load_kernel_parameter(new_params, d, kernel, 0);
    bta = exp(new_params(idx+1));
    
    model = [];
    model.params = new_params;
    model.ker_param = ker_param;
    model.bta = bta;
    
    %tr pred
    Sigma = 1/bta*eye(N) + ker_func(xTr,ker_param);
    Knn = ker_cross(xTr, xTr, ker_param);
    yTr_pred = Knn*(Sigma\yTr);
    model.yTr_pred = yTr_pred;
    
    %te pred
    if ~isempty(xTe)
        Ksn = ker_cross(xTe,xTr,ker_param);
        yTe_pred = Ksn*(Sigma\yTr);
        model.yTe_pred = yTe_pred;
        
        fTe_var = diag(ker_cross(xTe,xTe,ker_param)) - diag(Ksn*(Sigma\Ksn'));
        yTe_var = fTe_var + 1/bta;
        
        model.fTe_var = repmat(fTe_var,1,size(yTr,2));
        model.yTe_var = repmat(yTe_var,1,size(yTr,2));
    end
    
    model.forward = @forward;   %prediction function
    function [yTe_pred,fTe_var]=forward(xTe)
        Ksn = ker_cross(xTe,xTr,ker_param);
        yTe_pred = Ksn*(Sigma\yTr);
        fTe_var = diag(ker_cross(xTe,xTe,ker_param)) - diag(Ksn*(Sigma\Ksn'));
        yTe_var = fTe_var + 1/bta;
    end


    
end

