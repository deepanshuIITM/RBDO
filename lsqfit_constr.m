function p3 = lsqfit_constr(x,y)
%--------------------------------------------------------------------------
input = [ones(length(x),1) x x.^2 x.^3];
alpha0 = pinv(input)*y;

obj= @(alpha) sum((y- input*alpha).^2);

function [c,ceq] = constr(alpha)
c = [-alpha(4)-10^-5;((alpha(3))^2 - 3*alpha(2)*alpha(4)-10^-5)];
ceq = [];
end
optOptions = struct('MaxFunctionEvaluations', 100000);
p3 = fmincon(obj,alpha0,[],[],[],[],[],[],@constr,optOptions);
%--------------------------------------------------------------------------

end