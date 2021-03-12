function model = cigp_v2_03(xTr, yTr, xTe)
%train cigp model
% logg: v2_02. normalize data before runing
%       v2_03. different normalize data before runing x(:), y(:)
    
    %for x 
        %seperate each dim
            meanX = mean(xTr);
            stdX = std(xTr);
        %combine each dim
%             meanX = repmat(mean(xTr(:)), 1,size(xTr,2));
%             stdX = repmat(std(xTr(:)), 1,size(xTr,2));
        %normalize data
            xTr = (xTr - repmat(meanX,size(xTr,1),1) ) ./ (repmat(stdX,size(xTr,1),1) + eps);
            xTe = (xTe - repmat(meanX,size(xTe,1),1) ) ./ (repmat(stdX,size(xTe,1),1) + eps);

    %for y 
        %seperate each dim 
%             meanY = mean(yTr);
%             stdY = std(yTr);
        %combine each dim
            meanY = repmat(mean(yTr(:)), 1,size(yTr,2));
            stdY = repmat(std(yTr(:)), 1,size(yTr,2));
        %normalize data
            yTr = (yTr - repmat(meanY,size(yTr,1),1) ) ./ (repmat(stdY,size(yTr,1),1) + eps);

    %% add noise for model stability (prevent beta go to inf)
%     yTr = yTr + randn(size(yTr)) .* mean(yTr(:)) .* 0.001;
    
    %%

    a0 = 1e-3; b0 = 1e-3;
    [N,d] = size(xTr);

    m = size(yTr,2);
    D = yTr*yTr';
    assert(size(xTr,1)==size(yTr,1),'inconsistent data');
    
    log_bta = log(1/var(yTr(:)));
%     log_bta = log(1/eps);           %initilize with zero noise
    
    log_l = zeros(d,1);
    log_l = ones(d,1) * 1e-4;
    %log_l = 2*log(median(pdist(Xtr)))*ones(d,1);
    log_sigma = 0;
%     log_sigma0 = 0;
%     log_sigma = log(1e-4);
    log_sigma0 = log(1e-4);
    
%     log_sigma = log(eps);
%     log_sigma0 = log(eps);
    
    params = [log_l;log_sigma;log_sigma0;log_bta];
    fastDerivativeCheck(@(params) log_evidence_CIGP(params,a0,b0,m, xTr, D, 'ard'), params);    %pass check
    %max_iter = 1000;
    %new_param = minimize(param, @(param) log_evidence_lower_bound(param, x, y, m), max_iter);
    opt = [];
    opt.MaxIter = 100;
    opt.MaxFunEvals = 10000;
    new_params = minFunc(@(params) log_evidence_CIGP(params,a0,b0,m, xTr, D, 'ard'), params,opt);
    [ker_param,idx] = load_kernel_parameter(new_params, d, 'ard', 0);
    bta = exp(new_params(idx+1));
    
    model = [];
    model.params = new_params;
    model.ker_param = ker_param;
    model.bta = bta;
    
    %tr pred
    Sigma = 1/bta*eye(N) + ker_func(xTr,ker_param);
    Knn = ker_cross(xTr, xTr, ker_param);
    yTr_pred = Knn*(Sigma\yTr);
    
    %de-normalize
    yTr_pred = yTr_pred .* repmat(stdY,size(yTr_pred,1),1) + repmat(meanY,size(yTr_pred,1),1);
    model.yTr_pred = yTr_pred;
    
    %te pred
    if ~isempty(xTe)
        Ksn = ker_cross(xTe,xTr,ker_param);
        yTe_pred = Ksn*(Sigma\yTr);
        fTe_var = diag(ker_cross(xTe,xTe,ker_param)) - diag(Ksn*(Sigma\Ksn'));
        yTe_var = fTe_var + 1/bta;
        %de-normalize
        yTe_pred = yTe_pred .* repmat(stdY,size(yTe_pred,1),1) + repmat(meanY,size(yTe_pred,1),1);       
        model.yTe_pred = yTe_pred;
    
        model.fTe_var = repmat(fTe_var,1,size(yTr,2)) .* repmat(stdY,size(yTe_pred,1),1);
        model.yTe_var = repmat(yTe_var,1,size(yTr,2)) .* repmat(stdY,size(yTe_pred,1),1);
        
    end
    
%     xTr = (xTr - repmat(meanX,size(xTr,1)) ) ./ repmat(stdX,size(xTr,1));
%     yTr = (yTr - repmat(meanY,size(yTr,1)) ) ./ repmat(stdY,size(yTr,1));
%     xTe = (xTe - repmat(meanX,size(xTe,1)) ) ./ repmat(stdX,size(xTe,1));
    
   
    
    %     model.forward_mean = @(xTe) ker_cross(xTe,xTr,ker_param)*(Sigma\yTr);
    %     model.forward_var = @(xTe) diag(ker_cross(xTe,xTe,ker_param)) - diag(Ksn*(Sigma\Ksn'));
    
    model.forward = @forward;   %prediction function
    function [yTe_pred,fTe_var]=forward(xTe)
        
        xTe = (xTe - repmat(meanX,size(xTe,1),1) ) ./ (repmat(stdX,size(xTe,1),1) + eps);

        Ksn = ker_cross(xTe,xTr,ker_param);
        yTe_pred = Ksn*(Sigma\yTr);
        fTe_var = diag(ker_cross(xTe,xTe,ker_param)) - diag(Ksn*(Sigma\Ksn'));
        yTe_var = fTe_var + 1/bta;

        %de-normalize
        yTe_pred = yTe_pred .* repmat(stdY,size(yTe_pred,1),1) + repmat(meanY,size(yTe_pred,1),1);       
    %     model.yTe_pred = yTe_pred;

        fTe_var = repmat(fTe_var,1,size(yTr,2)) .* repmat(stdY,size(yTe_pred,1),1);
%         yTe_var = repmat(yTe_var,1,size(yTr,2)) .* repmat(stdY,size(yTe_pred,1),1);
    end
    
end
