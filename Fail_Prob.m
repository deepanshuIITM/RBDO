function beta = Fail_Prob(g)
frac = numel(find(g<0))/length(g);
CI = 0.5;

if(frac==0)
    %tail modeling
    g_sort = sort(g);
    g_50   = g_sort(1:CI*length(g));

    for i = 1:length(g_50)
        e_cdf(i,1) =  (i)/(length(g) + 1);
    end

    % fit CDF
    ln_TPNT = log(5+norminv(e_cdf));
    p3 = lsqfit_constr(g_50,ln_TPNT);
    beta = exp(p3(1)-5);
end

if(frac==1)
    %tail modeling
    g_sort = sort(g);
    g_50   = g_sort(1+CI*length(g):length(g));

    for i = 1:length(g_50)
        e_cdf(i,1) =  (length(g_50) + i)/(length(g) + 1);
    end

    % fit CDF
    ln_TPNT = log(5+norminv(e_cdf)); 
    p3 = lsqfit_constr(g_50,ln_TPNT);
    beta = exp(p3(1)-5);

else
    beta = -norminv(frac);
end