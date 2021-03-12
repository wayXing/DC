% cigp_DEMO
% demo file for conditional independnet GP 

clear 
addpath(genpath('../GP'));

%% date generate 
% define function
trueFunc = @(x) [exp(-x),sin(x),tanh(x)];

xtr = rand(10,1) * pi;
xte = linspace(0,1,100)' * pi;

ytr = trueFunc(xtr)
yte = trueFunc(xte);

% ytr = ytr + randn(size(ytr))*0.05;
% this ytr can be the coefficients extracted using a PCA or just the
% original high-dimensional raw data

%% use CIGP
% different version use different normalization for the data as
% pre-process. try different one for different application.

% try different method and plot

% model = cigp_v2(xtr, ytr, xte); % this cigp comes with no data normalization.
% model = cigp_v2_02(xtr, ytr, xte); % this cigp comes with seperate dimension normalization on x,y.
model = cigp_v2_03(xtr, ytr, xte); % this cigp comes with combine dimension normalization on y.

yPred = model.yTe_pred;
yPred_var = model.yTe_var; 

% test forward function
% [yPred,yPred_var] = model.forward(xte);


figure(1)
clf; hold on;
plot(xte, yPred,'--')
plot(xtr, ytr,'*')
plot(xte, yte,'-k')
hold off

%% use FIGP: fully independent GP

% yPred = [];
% for d = 1:3
%     model = cigp_v2(xtr, ytr(:,d), xte);
%     yPred = [yPred,model.yTe_pred];
% end 

%%
figure(2)
clf; hold on;
plot(xte, yPred,'--')
plot(xtr, ytr,'*')
plot(xte, yte,'-k')
hold off


figure(3)
clf; hold on;
% plot(xte, yPred,'--')
errorbar(repmat(xte,1,3),yPred, sqrt(yPred_var))
plot(xtr, ytr,'*')
plot(xte, yte,'-k')
hold off


figure(4)
clf; hold on;
% plot(xte, yPred,'--')
errorbar(repmat(xte,1,3),yPred, sqrt(yPred_var))
plot(xtr, ytr,'*')
plot(xte, yte,'-k')
hold off