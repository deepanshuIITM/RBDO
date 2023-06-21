clc
clear all
close all

%% Cantilever beam design problem
N1 = 196;
lb = [2 3];                                                  % lower bound and 
ub = [3 4];                                                  % upper bound

X_des = (lb + (ub-lb).*lhsdesign(N1,2));

%% Resampling 
physicalspace = [2 3; 3 4];                                  % Actual bound
lb = [2 3];                                                  % lower bound and 
ub = [3 4];                                                  % upper bound

normalizedspace = [zeros(1,2); ones(1,2)];   
% Normalized bound
normDOE = srgtsDOELHS(N1, 2, 10);
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
Nis = 200; 

Mu  = [500 1000 40000 2.9*10^7]; Sigma = [100 100 2*10^3 1.45*10^6];

for i=1:N
    sigma = 0.05*X_des(i,:);

    for a = 1:2       
        pd = makedist('Normal','mu',X_des(i,a),'sigma',sigma(a));
        tpd = truncate(pd,lb(a),ub(a));        
        X_s(:,a) = random(tpd,Nis,1);
        X_mcs(:,a) = random(tpd,100000,1);
    end
    
    for j = 1:4
        X_r(:,j) = normrnd(Mu(j),Sigma(j),Nis,1);
        X_r_mcs(:,j) = normrnd(Mu(j),Sigma(j),100000,1);
    end
    X = [X_s, X_r];
    X_mcs = [X_mcs, X_r_mcs];
    g_log = cant_g(X);
    g_mcs = cant_g(X_mcs);
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
    subplot(1,2,l); 
    histogram(beta_mcs(:,l),100)
    title(sprintf('beta_{mcs_%d}',l))

    figure(2)
    subplot(1,2,l); 
    histogram(beta(:,l),100)
    title(sprintf('beta_{%d}',l))
end

%% Fit Surrogate Model
for o = 1:2   
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
for o = 1:2
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
x0 = [2.5,3.5];
opts = optimoptions(@fmincon,'Algorithm','sqp','Display', 'iter');

[PRS_x, PRS_f, PRS_exf] = fmincon(@area,x0,[],[],[],[],lb,ub,@nlcon_rel,opts); %,@nlcon_rel

[PRS_mcs_x, PRS_mcs_f, PRS_mcs_exf] = fmincon(@area,x0,[],[],[],[],lb,ub,@nlcon_rel_mcs,opts); %,@nlcon_rel

%% Constraint function
function g = cant_g(x)
L = 100; D_0 = 2.5;

w = x(:,1); t = x(:,2); X = x(:,3); Y = x(:,4); sigma_y = x(:,5); E = x(:,6);  

g1 = sigma_y - (6*L./(w.*t)).*(X./w + Y./t);

g2 = D_0 - (4*(L^3)./(E.*w.*t)).*sqrt((X./(w.^2)).^2 + (Y./(t.^2)).^2);
g = [g1,g2];

end

%% Reliability constraints
function [c, ceq] = nlcon_rel(x)
load('canti_PRS.mat')
beta_t = 4.0;
rel1 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_num2str(1));
rel2 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_num2str(2));
c = [rel1; rel2];
ceq = [];
end

%% Reliability constraints
function [c, ceq] = nlcon_rel_mcs(x)
load('canti_PRS_mcs.mat')
beta_t = 4.0;
rel1 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_MCS_num2str(1));
rel2 = beta_t - srgtsPRSEvaluate(x, srgtSRGT_PRS_MCS_num2str(2));
c = [rel1; rel2];
ceq = [];
end

%% Weight calculation
function A = area(x)
w = x(:,1); t = x(:,2);
A =w.*t;
end
