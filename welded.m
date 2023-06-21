clc
clear all
close all

%% Welded beam design problem
N_des = 184;
normDOE = srgtsDOELHS(N_des, 4, 5);
physicalspace = [0.125,0.1,0.1,0.125; 5,10,10,5];                                                       % Actual bound
lb = [0.125,0.1,0.1,0.125];                                                                             % lower bound and 
ub = [5,10,10,5];                                                                                       % upper bound
normalizedspace = [zeros(1,4); ones(1,4)]; 

%% Resampling 
physicalspace = [0.125,0.1,0.1,0.125; 5,10,10,5];                                 % Actual bound
lb = [0.125,0.1,0.1,0.125];                                                       % lower bound and 
ub = [5,10,10,5];                                                                 % upper bound
X1 = srgtsScaleVariable(normDOE, normalizedspace, physicalspace); 

mm = [lb; ub]';                                                                         % 2D
nd = size(mm,1);                                                                        % number of dimensions
mmC = mat2cell(mm,ones(1,nd),2);
[p{1:nd}] = ndgrid(mmC{:});     
P = reshape(cat(nd,p{:}),[],nd);

X_des = [X1; P];                                                                             

%% perform FEA to compute the response for each combination of design
% variables
N = size(X_des,1);
Nis = 500; 

for k=1:N
    sigma = [0.01, 0.1, 0.1, 0.01];
    for a = 1:4       
        pd = makedist('Normal','mu',X_des(k,a),'sigma',sigma(a));
        tpd = truncate(pd,lb(a),ub(a));        
        X_s(:,a) = random(tpd,Nis,1);
        X_mcs(:,a) = random(tpd,100000,1);
    end

    g_log = weld_g(X_s);
    g_mcs = weld_g(X_mcs);
    % Reliability index estimation
    for o = 1:size(g_log,2)
        beta(k,o) = Fail_Prob(g_log(:,o));
        
        Pf(k,o) = numel(find(g_mcs(:,o)<0))/size(g_mcs,1);
        beta_mcs(k,o) = -norminv(Pf(k,o));
    end
    
end

%% Reliability estimates
beta_mcs(beta_mcs>3) = 4;
beta_mcs(beta_mcs<1) = 2;

beta(beta>3) = 4;
beta(beta<1) = 2;

% beta_mcs(beta_mcs==Inf) = max(beta_mcs(isfinite(beta_mcs)))+1;
% beta_mcs(beta_mcs==-Inf) = min(beta_mcs(isfinite(beta_mcs)))-1;
% 
% beta(beta==Inf) = max(beta(isfinite(beta)))+1;
% beta(beta==-Inf) = min(beta(isfinite(beta)))-1;

%% Plot Histogram

for l = 1:size(g_log,2)
    figure(1)
    subplot(2,3,l); 
    histogram(beta_mcs(:,l),100)
    title(sprintf('beta_{mcs_%d}',l))

    figure(2)
    subplot(2,3,l); 
    histogram(beta(:,l),100)
    title(sprintf('beta_{%d}',l))
end

%% Fit Surrogate Model
for t = 1:size(g_log,2)   
    [srgtSRGT_PRS_num2str(t), PRESSRMS_PRS(t), eXV_PRS(:,t), srgtOPT_PRS_num2str(t)] = build_PRS_SRGT(X_des,beta(:,t));
    [srgtSRGT_PRS_MCS_num2str(t), PRESSRMS_PRS_MCS(t), eXV_PRS_MCS(:,t), srgtOPT_PRS_MCS_num2str(t)] = build_PRS_SRGT(X_des,beta_mcs(:,t));

    
%     [srgtSRGT_KRG_num2str(o), PRESSRMS_KRG(o), eXV_KRG(:,o), srgtOPT_KRG_num2str(o)] = build_KRG_SRGT(X_des,beta(:,o));
% 
%     
%     [srgtSRGT_RBF_num2str(o), PRESSRMS_RBF(o), eXV_RBF(:,o), srgtOPT_RBF_num2str(o)] = build_RBF_SRGT(X_des,beta(:,o));
%     
%     [srgtSRGT_WAS_num2str(o), PRESSRMS_WAS(o)] = build_WAS_SRGT(X_des,...
%        srgtSRGT_PRS_num2str(o), srgtOPT_PRS_num2str(o), eXV_PRS(:,o), ...
%        srgtSRGT_KRG_num2str(o), srgtOPT_KRG_num2str(o), eXV_KRG(:,o), ...
%        srgtSRGT_RBF_num2str(o), srgtOPT_RBF_num2str(o), eXV_RBF(:,o));
end

%%
for o = 1:size(g_log,2)
    rng_err(o) = [PRESSRMS_PRS(o)]/range(beta(:,o));
    mu_err(o)  = [PRESSRMS_PRS(o)]/mean(beta(:,o));
    rng_mcs_err(o) = [PRESSRMS_PRS_MCS(o)]/range(beta_mcs(:,o));
    mu_mcs_err(o)  = [PRESSRMS_PRS_MCS(o)]/mean(beta_mcs(:,o));
 %   rng_err(o,1:4) = [PRESSRMS_PRS(o),PRESSRMS_KRG(o),PRESSRMS_RBF(o),PRESSRMS_WAS(o)]/range(beta(:,o));
 %   mu_err(o,1:4)  = [PRESSRMS_PRS(o),PRESSRMS_KRG(o),PRESSRMS_RBF(o),PRESSRMS_WAS(o)]/mean(beta(:,o));
end

%% pred R^2
for o = 1:size(g_log,2)
    ybar = mean(beta(:,o));
    ybar_mcs = mean(beta_mcs(:,o));
    pred_R2(o) = 1-200*[PRESSRMS_PRS(o)]/((exp(beta(:,o))-ybar)'*(exp(beta(:,o))-ybar));
    pred_R2_mcs(o)  = 1-200*[PRESSRMS_PRS_MCS(o)]/((exp(beta_mcs(:,o))-ybar_mcs)'*(exp(beta_mcs(:,o))-ybar_mcs));
end

%% Optimization
x0 = 0.5*(lb+ub);
opts = optimoptions(@fmincon,'Algorithm','sqp','Display', 'iter');

[PRS_x, PRS_f, PRS_exf] = fmincon(@weld_cost,x0,[],[],[],[],lb,ub,@nlcon_rel,opts); %,@nlcon_rel

[PRS_mcs_x, PRS_mcs_f, PRS_mcs_exf] = fmincon(@weld_cost,x0,[],[],[],[],lb,ub,@nlcon_rel_mcs,opts); %,@nlcon_rel

%%
weld_g(PRS_x)
weld_g(PRS_mcs_x)

%% constraint function
function [g] = weld_g(x)
x1 = x(:,1); x2 = x(:,2); x3 = x(:,3); x4 = x(:,4);
P = 6000; L=14;
g1 = x1-x4;
tau1 = 1./(sqrt(2)*x1.*x2); R = sqrt((x1+x3).^2 + x2.^2); tau2 = (L+0.5*x2).*R./(sqrt(2)*x1.*x3.*(0.33*x2.^2 +(x1+x3).^2));
tau = P*sqrt(tau1.^2 + tau2.^2 + 2.*tau1.*tau2.*x2./R);
sigma = 6*P*L./(x4.*x3.^2);
g2 = tau-13600;
g3 = sigma-30000;
g4 = 6000-64746.022*(1-0.0282346*x3).*x3.*(x4.^3);
g5 = 0.25-((0.36587*6)./(x4.*x3.^3));
g=[g1 g2 g3 g4 g5];
end

%% objective function
function cost = weld_cost(x)
x1 = x(:,1); x2 = x(:,2); x3 = x(:,3); x4 = x(:,4);
cost = 1.10471*(x1.^2).*x2 + 0.048111*x3.*x4.*(14+x2);
end

%% Reliability constraints for logTPNT
function [c, ceq] = nlcon_rel(x)
load('weld_PRS.mat')
beta_t = 3.0;
rel1 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_num2str(1));
rel2 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_num2str(2));
rel3 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_num2str(3));
rel4 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_num2str(4));
rel5 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_num2str(5));
c = [rel1; rel2; rel3; rel4];% rel5];
ceq = [];
end

%% Reliability constraints for MCS
function [c, ceq] = nlcon_rel_mcs(x)
load('weld_PRS_MCS.mat')
beta_t = 3.0;
rel1 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_MCS_num2str(1));
rel2 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_MCS_num2str(2));
rel3 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_MCS_num2str(3));
rel4 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_MCS_num2str(4));
rel5 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_MCS_num2str(5));
c = [rel1; rel2; rel3; rel4];% rel5];
ceq = [];
end