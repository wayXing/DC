function [ypred, model] = dc(xtr,Ytr,xte,r)
% DC function
% 
% Inputs:
% xtr - [N_train x dim_x] matrix, input parameters
% Ytr - [1 x N_fidelity] cell, each element contains the corresponding output to
%       xtr and has to be a [N_train x dim_y] matrix. Note that not all data
%       would be used for training
% xte - [N_test x dim_x] matrix, testing inputs 
% r   - scalar value, number of residual basis for each fidelity (must be
%       interal) 
%       OR preserved energy for each fidelity (must >0 & <=1)
% 
% Outputs:
% yPred - predictions for xte at the highest-fidelity
% model - model info
% 
% Author: Wei W. Xing 
% email address: wayne.xingle@gmail.com
% Last revision: 12-March-2021


iMethod_dgp = 3;    %different deep GP types. see dgp_func.m
dgp_func = @deepGp_v01_04;   % dgp with normalize y(:)

Ypred = [];
% try 
    [Ztr,model_mfrPCA] = mfrPCA_v02(Ytr,r);
%     [Ztr,model_mfrPCA] = mfrPCA(Ytr,r);
    
    [Zpred, model_dgp] = dgp_func(xtr,Ztr,xte, iMethod_dgp);
    Ypred_i = model_mfrPCA.recover(Zpred);

    ypred = Ypred_i{end};

    model.model_dgp = model_dgp;
    model.model_mfrPCA = model_mfrPCA;
    
% catch 
%     ypred = [];
%     model = [];
% end


