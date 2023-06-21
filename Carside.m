clc
clear all
close all

%% Carside Impact Problem
N_des = 72;
normDOE = srgtsDOELHS(N_des, 7, 5);

%% Resampling 
physicalspace = [0.5 0.45 0.5 0.5 0.875 0.4 0.4; 1.5 1.35 1.5 1.5 2.625 1.2 1.2];        % Actual bound
lb = [0.5 0.45 0.5 0.5 0.875 0.4 0.4];                                                  % lower bound and 
ub = [1.5 1.35 1.5 1.5 2.625 1.2 1.2];                                                  % upper bound

normalizedspace = [zeros(1,7); ones(1,7)];   
% Normalized bound
X1 = srgtsScaleVariable(normDOE, normalizedspace, physicalspace); 

mm = [lb; ub]'; % 10D
nd = size(mm,1) ; % number of dimensions
mmC = mat2cell(mm,ones(1,nd),2);
[p{1:nd}] = ndgrid(mmC{:});
P = reshape(cat(nd,p{:}),[],nd);

X_des = [X1; P]; %X = X(180:end,:);

%% perform FEA to compute the response for each combination of design variables
N = size(X_des,1);
Nis = 500; 

Mu  = [500 1000 40000 2.9*10^7]; Sigma = [100 100 2*10^3 1.45*10^6];

for i=1:N
        sigma = [0.03 0.03 0.03 0.03 0.05 0.03 0.03 0.006 0.006 10 10];
    for a = 1:7       
        pd = makedist('Normal','mu',X_des(i,a),'sigma',sigma(a));
        tpd = truncate(pd,lb(a),ub(a));        
        X_s(:,a) = random(tpd,Nis,1);
        X_mcs(:,a) = random(tpd,100000,1);
    end
    Mu  = [0.345 0.192 0 0]; Sigma = [0.006 0.006 10 10];
    for j = 1:4
        X_r(:,j) = normrnd(Mu(j),Sigma(j),Nis,1);
        X_r_mcs(:,j) = normrnd(Mu(j),Sigma(j),100000,1);
    end
    X = [X_s, X_r];
    X_mcs = [X_mcs, X_r_mcs];
    g_log = crash_worth_g(X);
    g_mcs = crash_worth_g(X_mcs);
    % Reliability index estimation
    for o = 1:size(g_log,2)
        beta(i,o) = Fail_Prob(g_log(:,o));
        
        Pf(i,o) = numel(find(g_mcs(:,o)<0))/size(g_mcs,1);
        beta_mcs(i,o) = -norminv(Pf(i,o));
    end
    
end

%% Reliability estimates
beta_mcs(beta_mcs==Inf) = max(beta_mcs(isfinite(beta_mcs)))+1;
beta_mcs(beta_mcs==-Inf) = min(beta_mcs(isfinite(beta_mcs)))-1;

beta(beta==Inf) = max(beta(isfinite(beta)))+1;
beta(beta==-Inf) = min(beta(isfinite(beta)))-1;

%% Plot Histogram

for l = 1:size(g_log,2)
    figure(1)
    subplot(2,5,l); 
    histogram(beta_mcs(:,l),100)
    title(sprintf('beta_{mcs_%d}',l))

    figure(2)
    subplot(2,5,l); 
    histogram(beta(:,l),100)
    title(sprintf('beta_{%d}',l))
end

%% Fit Surrogate Model
for o = 1:10  
    [srgtSRGT_PRS_num2str(o), PRESSRMS_PRS(o), eXV_PRS(:,o), srgtOPT_PRS_num2str(o)] = build_PRS_SRGT(X_des,beta(:,o));
    [srgtSRGT_PRS_MCS_num2str(o), PRESSRMS_PRS_MCS(o), eXV_PRS_MCS(:,o), srgtOPT_PRS_MCS_num2str(o)] = build_PRS_SRGT(X_des,beta_mcs(:,o));

    
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
for o = 1:10
    rng_err(o,1:4) = [PRESSRMS_PRS(o)]/range(beta(:,o));
    mu_err(o,1:4)  = [PRESSRMS_PRS(o)]/mean(beta(:,o));
    rng_mcs_err(o,1:4) = [PRESSRMS_PRS_MCS(o)]/range(beta_mcs(:,o));
    mu_mcs_err(o,1:4)  = [PRESSRMS_PRS_MCS(o)]/mean(beta_mcs(:,o));
 %   rng_err(o,1:4) = [PRESSRMS_PRS(o),PRESSRMS_KRG(o),PRESSRMS_RBF(o),PRESSRMS_WAS(o)]/range(beta(:,o));
 %   mu_err(o,1:4)  = [PRESSRMS_PRS(o),PRESSRMS_KRG(o),PRESSRMS_RBF(o),PRESSRMS_WAS(o)]/mean(beta(:,o));
end

%% pred R^2
for o = 1:size(g_log,2)
    ybar = mean(beta(:,o));
    ybar_mcs = mean(beta_mcs(:,o));
    pred_R2(o) = 1-200*[PRESSRMS_PRS(o)]/((beta(:,o)-ybar)'*(beta(:,o)-ybar));
    pred_R2_mcs(o)  = 1-200*[PRESSRMS_PRS_MCS(o)]/((beta_mcs(:,o)-ybar_mcs)'*(beta_mcs(:,o)-ybar_mcs));
end

%% Optimization
x0 = 0.5*(lb + ub);
opts = optimoptions(@fmincon,'Algorithm','sqp','Display', 'iter');

[PRS_x, PRS_f, PRS_exf] = fmincon(@weight,x0,[],[],[],[],lb,ub,@nlcon_rel,opts); %,@nlcon_rel

[PRS_mcs_x, PRS_mcs_f, PRS_mcs_exf] = fmincon(@weight,x0,[],[],[],[],lb,ub,@nlcon_rel_mcs,opts); %,@nlcon_rel

%%
nlcon_rel(x0)

%% Objective function 
function W = weight(x)
x1 = x(:,1); x2 = x(:,2); x3 = x(:,3); x4 = x(:,4); x5 = x(:,5);  x6 = x(:,6); x7 = x(:,7);
W= 1.98 + 4.90*x1+ 6.67*x2 + 6.98*x3 + 4.01*x4 + 1.78*x5 + 2.73*x7;
end

%% Reliability constraint for logTPNT
function [c_IS, ceq_IS] = nlcon_rel(x)
    load('carside_PRS.mat')
    cons_IS = zeros(1,10);
    beta_t = 3.0;
    for t= 1:10
        cons_IS(1,t) = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_num2str(t));
    end
        c_IS = cons_IS;
        ceq_IS = [];
end

%% Reliability constraint for MCS
function [c_IS, ceq_IS] = nlcon_rel_mcs(x)
    load('carside_PRS_MCS.mat')
    cons_IS = zeros(1,10);
    beta_t = 3.0;
    for t= 1:10
        cons_IS(1,t) = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_MCS_num2str(t));
    end
        c_IS = cons_IS;
        ceq_IS = [];
end