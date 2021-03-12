function [Z,model] = mfrPCA_v02(Y,r)
% multi fidelity residual PCA
% logg: v_02: use pcaInit_v02.(z is not scaled to normal)
% 
% Inputs:
% Y - [1 x N_fidelity] cell, each element contains the corresponding output to
%       xtr and has to be a [N_train x dim_y] matrix. Note that not all data
%       would be used for training
% r - scalar value, number of residual basis for each fidelity (must be
%     interal) 
%     OR preserved energy for each fidelity (must >0 & <=1)
% 
% Outputs:
% Z - [1 x N_fidelity] coefficients. Each is a [N x r] matrix
% model - model info
% 
% Author: Wei W. Xing 
% email address: wayne.xingle@gmail.com
% Last revision: 12-March-2021


assert(iscell(Y),'Y must be cell');

Z=[];
nLv = length(Y);

k=1;
[Z{k}, Pca_Models{k}] = pcaInit_v02(Y{k}, r);

for k = 2:nLv 
    nSample_k = size(Y{k}, 1);
    residual_k = Y{k} - Y{k-1}(1:nSample_k,:);
    [Z{k}, Pca_Models{k}] = pcaInit_v02(residual_k, r);
end

model.Pca_Models = Pca_Models;
model.recover = @(Zte) recover(Zte,Pca_Models);
model.project = @(Yte) project(Yte,Pca_Models);
end

function Zte = project(Yte,Models)
k=1;
Zte{k} = Models{k}.project(Yte{k});
for k = 2:length(Yte) 
    nSample_k = size(Yte{k}, 1);
    residual_k = Yte{k} - Yte{k-1}(1:nSample_k,:);
    Zte{k} = Models{k}.project(residual_k);
end

end

function Yte = recover(Zte,Models)
    for k = 1:length(Zte)
        Yte{k} = Models{k}.recover(Zte{k});
    end
    for k = 2:length(Zte)
        Yte{k} = Yte{k}+Yte{k-1};
    end
end