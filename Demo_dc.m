% Demo_dc

%% load data
clear
rng(1234)
% addpath(genpath('./../../code'))
dataName = 'syn_v02';
load(dataName)

%% prepare data
Ytr{1} = Y{1}(1:256,:);     %fidelity-1 training
Ytr{2} = Y{2}(1:128,:);     %fidelity-2 training
Ytr{3} = Y{3}(1:64,:);
xtr = X(1:256,:);

Yte{1} = Y{1}(257:512,:);   %fidelity-1 testing
Yte{2} = Y{2}(257:512,:);   %fidelity-2 testing
Yte{3} = Y{3}(257:512,:);

xte = X(257:512,:);
yte = Yte{end};
%%
% r = 5;      %number of residual bases
% [ypred, model] = dc(xtr,Ytr,xte,r);

%% prediction error with increasing number of bases
err = [];
i=1;
for r = 2:2:8
    [ypred, model] = dc(xtr,Ytr,xte,r);
    
    err(i) = sum((ypred(:) - yte(:)).^2);
    err(i) = sqrt(err(i)/sum(yte(:).^2));
    i = i + 1;
end

figure(1)
plot([2:2:8],err,'o-','LineWidth',2);
ylabel('Relative Error')
xlabel('#Bases')

%% prediction error with increasing preserved energy
err = [];
i=1;
for r = [0.8,0.9,0.99,0.999]
    [ypred, model] = dc(xtr,Ytr,xte,r);
    
    err(i) = sum((ypred(:) - yte(:)).^2);
    err(i) = sqrt(err(i)/sum(yte(:).^2));
    i = i + 1;
end

figure(2)
plot([0.8,0.9,0.99,0.999],err,'o-','LineWidth',2);
ylabel('Relative Error')
xlabel('#Bases')

