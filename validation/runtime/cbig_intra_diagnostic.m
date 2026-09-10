function [Params, Diagnostic] = cbig_intra_diagnostic(Params,setting_params)

% Intra-subject variability level

flag_psi = zeros(setting_params.num_sub, 1);
stop_intra = 0;
iter_intra = 0;
while(stop_intra == 0)
    iter_intra = iter_intra + 1;
    if iter_intra > 10000
        Diagnostic.converged = false;
        Diagnostic.iterations = iter_intra-1;
        return;
    end
    fprintf('It is inter interation %d intra iteration %d..update s_psi and sigma..\n',...
        Params.iter_inter,iter_intra);
    % update s_psi
    s_psi_update = CBIG_nansum(bsxfun(@times,Params.s_t_nu,repmat(Params.sigma,size(Params.s_t_nu,1), ...
                   1,size(Params.s_t_nu,3),size(Params.s_t_nu,4))),3);
    s_psi_update = reshape(s_psi_update,...
                   size(s_psi_update,1),size(s_psi_update,2),...
                   size(s_psi_update,3)*size(s_psi_update,4));
    s_psi_update = bsxfun(@plus,s_psi_update,bsxfun(@times,Params.epsil,Params.mu));
    s_psi_update = bsxfun(@times,s_psi_update,1./sqrt(sum((s_psi_update).^2)));
   
    for s = 1:setting_params.num_sub
        checkpsi = diag(s_psi_update(:,:,s)'*Params.s_psi(:,:,s));
        checkpsi_flag = (sum(1-checkpsi < setting_params.epsilon) < setting_params.num_clusters);
        if(checkpsi_flag < 1)
            flag_psi(s,1) = 1;
        end
    end
    Params.s_psi = s_psi_update;
    
    % update sigma
    sigma_update = bsxfun(@times,Params.s_psi,permute(Params.s_t_nu,[1,2,4,3]));
    sigma_update = CBIG_nanmean(mean(sum(sigma_update,1),3),4);%1xLxSxT=>1xL
    for i = 1:setting_params.num_clusters
        sigma_update(i) = invAd(setting_params.dim,sigma_update(i));
    end
    
    if((sum(flag_psi) == setting_params.num_sub) && (mean(abs(Params.sigma-sigma_update)./Params.sigma) ...
       < setting_params.epsilon))
        stop_intra = 1;
    end
    Diagnostic.relative_sigma_change = mean(abs(Params.sigma-sigma_update)./Params.sigma);
    Params.sigma = sigma_update;
end
Diagnostic.converged = true;
Diagnostic.iterations = iter_intra;
end

function out = Ad(in,D)
out = besseli(D/2,in) ./ besseli(D/2-1,in);
end

function out = Cdln(k,d,k0)
k = double(k);

% Computes the logarithm of the partition function of vonMises-Fisher as
% a function of kappa

sizek = size(k);
k = k(:);

out = (d/2-1).*log(k)-log(besseli((d/2-1)*ones(size(k)),k));
if(d<1200)
    k0 = 500;
elseif(d>=1200 && d<1800)
    k0 = 650;
else
    error('dimension is too high, need  to specify k0');
end
fk0 = (d/2-1).*log(k0)-log(besseli(d/2-1,k0));
nGrids = 1000;

maskof = find(k>k0);
nkof = length(maskof);

% The kappa values higher than the overflow

if nkof > 0

    kof = k(maskof);

    ofintv = (kof - k0)/nGrids;
    tempcnt = (1:nGrids) - 0.5;
    ks = k0 + repmat(tempcnt,nkof,1).*repmat(ofintv,1,nGrids);
    adsum = sum( 1./((0.5*(d-1)./ks) + sqrt(1+(0.5*(d-1)./ks).^2)) ,2);

    out(maskof) =  fk0 - ofintv .* adsum;

end

out = single(reshape(out,sizek));
end

function [outu] = invAd(D,rbar)

rbar = double(rbar);

outu = (D-1).*rbar./(1-rbar.^2) + D/(D-1).*rbar;

[i] = besseli(D/2-1,outu);


if ((i == Inf)||(isnan(i)) || (i==0))
    out = outu - D/(D-1)*rbar/2;
    exitflag = Inf;
else
    [outNew, fval exitflag]  = octave_fzero(@(argum) Ad(argum,D)-rbar,outu);
    if exitflag == 1
        out = outNew;
    else
        out = outu - D/(D-1)*rbar/2;
    end
end
end

