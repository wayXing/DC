function model = cigp_cat_v03(cigp_model, xTr, yTr, xTe)
% cat new input to cigp model. The cigp_model is model gnerated by
% cigp_v2_x.
% the function inherit hyp from cigp_model and retrain the model

meanX = mean(xTr);
stdX = std(xTr);
%combine each dim
%             meanX = repmat(mean(xTr(:)), 1,size(xTr,2));
%             stdX = repmat(std(xTr(:)), 1,size(xTr,2));
%normalize data
xTr = (xTr - repmat(meanX,size(xTr,1),1) ) ./ repmat(stdX,size(xTr,1),1);
xTe = (xTe - repmat(meanX,size(xTe,1),1) ) ./ repmat(stdX,size(xTe,1),1);

%for y 
%seperate each dim 
%             meanY = mean(yTr);
%             stdY = std(yTr);
%combine each dim
meanY = repmat(mean(yTr(:)), 1,size(yTr,2));
stdY = repmat(std(yTr(:)), 1,size(yTr,2));
%normalize data
yTr = (yTr - repmat(meanY,size(yTr,1),1) ) ./ repmat(stdY,size(yTr,1),1);

cigp_model_l_dim = length(cigp_model.ker_param.l);


a0 = 1e-3; b0 = 1e-3;
[N,d] = size(xTr);

m = size(yTr,2);
D = yTr*yTr';
assert(size(xTr,1)==size(yTr,1),'inconsistent data');
% log_bta = log(1/var(yTr(:)));
% log_l = zeros(d,1);
%log_l = 2*log(median(pdist(Xtr)))*ones(d,1);

log_bta = log(cigp_model.bta);
log_sigma = log(cigp_model.ker_param.sigma);
log_sigma0 = log(cigp_model.ker_param.sigma0);
log_l = log([cigp_model.ker_param.l;ones(d-cigp_model_l_dim,1)*0.01]);

params = [log_l;log_sigma;log_sigma0;log_bta];
fastDerivativeCheck(@(params) log_evidence_CIGP(params,a0,b0,m, xTr, D, 'ard'), params);
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
    %de-normalize
    yTe_pred = yTe_pred .* repmat(stdY,size(yTe_pred,1),1) + repmat(meanY,size(yTe_pred,1),1);

    model.yTe_pred = yTe_pred;
end






end