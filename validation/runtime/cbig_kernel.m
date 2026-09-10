function Params = cbig_kernel(data, g_mu, max_iter)
% Adapted infrastructure only; equations from CBIG, see ../upstream/LICENSE-CBIG.md.
setting_params.num_sub = size(data.series, 3);
setting_params.num_session = size(data.series, 4);
setting_params.num_clusters = size(g_mu, 2);
setting_params.dim = size(data.series, 2) - 1;
setting_params.ini_concentration = 500;
setting_params.epsilon = 1e-4;
setting_params.conv_th = 1e-5;
setting_params.max_iter = max_iter;
setting_params.g_mu = g_mu;
% mu: DxL. The group-level functional connectivity profiles of networks
Params.mu = setting_params.g_mu;

% epsil: 1xL. The inter-subject concentration parameter, which represents
% inter-subject functional connectivity variability. A large epsil_l 
% indicats low inter-subject functional connectivity variability for 
% network l.
Params.epsil = setting_params.ini_concentration*ones(1, setting_params.num_clusters);

% s_psi: DxLxS. The functional connectivity profiles of L networks for S
% subjects.
Params.s_psi = repmat(setting_params.g_mu, 1, 1, setting_params.num_sub);

% sigma: 1xL. The intra-subject concentration parameter, which represents
% intra-subject functional connectivity variability. A large sigma_l
% indicates low intra-subject functional connectivity variability for
% network l.
Params.sigma = setting_params.ini_concentration*ones(1, setting_params.num_clusters);

% s_t_nu: DxLxTxS. The functional connectivity profiles of L networks for S
% subjects and each subject has T sessions.
Params.s_t_nu = repmat(setting_params.g_mu, 1, 1, setting_params.num_session, setting_params.num_sub);

% kappa: 1xL. The inter-region concentration parameter, which represents
% inter-region functional connectivity variability. A large kappa_l 
% indicates low inter-region functional variability for network l. However,
% please note in this script, we assume kappa to be the same across 
% networks.
Params.kappa = setting_params.ini_concentration*ones(1, setting_params.num_clusters);

% s_lambda: NxLxS. The posterior probability of the individual-specific
% parcellation of each subject. 
log_vmf = permute(Params.s_t_nu, [1,2,4,3]);
log_vmf = mtimesx(data.series, log_vmf);%NxLxSxT
log_vmf = bsxfun(@times, permute(log_vmf,[2,1,3,4]), transpose(Params.kappa));%LxNxSxT
log_vmf = bsxfun(@plus,Cdln(transpose(Params.kappa), setting_params.dim), log_vmf);%LxNxSxT
log_vmf = CBIG_nansum(log_vmf, 4);
log_vmf = permute(log_vmf, [2,1,3]);
s_lambda = bsxfun(@minus, log_vmf, max(log_vmf,[],2));
mask = repmat((sum(s_lambda,2)==0), 1, setting_params.num_clusters, 1);
s_lambda = exp(s_lambda);
Params.s_lambda = bsxfun(@times, s_lambda, 1./sum(s_lambda,2));
Params.s_lambda(mask) = 0;

%theta: NxL. The spatial prior denotes the probability of networks
%occurring at each spatial location.
Params.theta = mean(Params.s_lambda, 3);
theta_num = sum(Params.s_lambda ~= 0, 3);
Params.theta(Params.theta ~= 0) = Params.theta(Params.theta ~= 0)./theta_num(Params.theta ~= 0);


%% EM

stop_inter = 0;
cost_inter = 0;
Params.iter_inter = 0;
while(stop_inter == 0)
    Params.iter_inter = Params.iter_inter + 1;
    %% Intra subject variability
    cost = 0;
    iter_intra_em = 0;
    stop_intra_em = 0;
    Params.sigma = setting_params.ini_concentration*ones(1, setting_params.num_clusters);
    Params.s_psi = repmat(setting_params.g_mu, 1, 1, setting_params.num_sub);
    while(stop_intra_em == 0)
        iter_intra_em = iter_intra_em + 1;

        fprintf('Inter-region iteration %d ...\n', iter_intra_em);
        Params.kappa = setting_params.ini_concentration*ones(1, setting_params.num_clusters);
        Params.s_t_nu = repmat(setting_params.g_mu, 1, 1, setting_params.num_session, setting_params.num_sub);
        Params = vmf_clustering_subject_session(Params, setting_params, data);

        fprintf('Intra-subject variability level...\n');
        Params = intra_subject_var(Params, setting_params);

        update_cost = bsxfun(@times, Params.s_psi, permute(Params.s_t_nu, [1,2,4,3]));
        update_cost = sum(bsxfun(@times, Params.sigma, update_cost), 1);
        update_cost = bsxfun(@plus, Cdln(Params.sigma, setting_params.dim), update_cost);
        update_cost = CBIG_nansum(sum(sum(update_cost, 2), 3), 4) ...
                    + sum(sum(bsxfun(@plus, ...
                      sum(bsxfun(@times, bsxfun(@times, Params.mu, Params.s_psi),Params.epsil), 1), ...
                      Cdln(Params.epsil, setting_params.dim)), 2), 3);
        update_cost = update_cost + sum(Params.cost_em);
        if(abs(abs(update_cost - cost)./cost) <= setting_params.epsilon)
            stop_intra_em = 1;
            Params.cost_intra = update_cost;
        end
        if(iter_intra_em >= 50)
            stop_intra_em = 1;
            Params.cost_intra = update_cost;
        end
        cost = update_cost;
    end

    %% Inter subject variability
    fprintf('Inter-subject variability level ...\n');
    Params = inter_subject_var(Params, setting_params);

    update_cost_inter = Params.cost_intra;    
    Params.Record(Params.iter_inter) = update_cost_inter;
    if(abs(abs(update_cost_inter - cost_inter)./cost_inter) <= setting_params.conv_th || ...
            Params.iter_inter >= setting_params.max_iter)
        stop_inter = 1;
        Params.cost_inter = update_cost_inter;

    end
    cost_inter = update_cost_inter;
end

end


function Params=inter_subject_var(Params,setting_params)

% Inter-subject variability level

% update mu
mu_update = sum(Params.s_psi, 3);
mu_update = bsxfun(@times, mu_update, 1./sqrt(sum((mu_update).^2)));

% update epsil
epsil_update = bsxfun(@times, Params.s_psi, mu_update);
epsil_update = sum(sum(epsil_update, 1), 3);
epsil_update = epsil_update./setting_params.num_sub;
for i = 1:setting_params.num_clusters
    epsil_update(i) = invAd(setting_params.dim, epsil_update(i));
    if(epsil_update(i) < setting_params.ini_concentration)
        epsil_update(i) = setting_params.ini_concentration;
        fprintf('[WARNING] epsil of %d is less than the minimal value %d \n',...
            i, setting_params.ini_concentration);
    end
    if(isinf(epsil_update(i)))
        epsil_update(i) = Params.epsil(i);
        fprintf('[WARNING] epsil of %d is Inf\n',i);
    end
end
Params.mu = mu_update;
Params.epsil = epsil_update;
end

function Params = intra_subject_var(Params,setting_params)

% Intra-subject variability level

flag_psi = zeros(setting_params.num_sub, 1);
stop_intra = 0;
iter_intra = 0;
while(stop_intra == 0)
    iter_intra = iter_intra + 1;
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
    Params.sigma = sigma_update;
end
end

function Params = vmf_clustering_subject_session(Params,setting_params,data)

% Inter-region level

stop_em = 0;
iter_em = 0;
cost = zeros(1, setting_params.num_sub);

while(stop_em == 0)
    iter_em = iter_em + 1;
    fprintf('It is EM iteration.. %d..\n',iter_em);
    %% Mstep
    flag_nu = zeros(setting_params.num_sub,setting_params.num_session);
    stop_m = 0;
    iter_m = 0;
    fprintf('M-step..\n');
    while(stop_m == 0)
        iter_m = iter_m + 1;
        
        % update kappa
        s_lambda = Params.s_lambda;%NxLxS
        kappa_update = mtimesx(data.series,permute(Params.s_t_nu,[1,2,4,3]));
        kappa_update = bsxfun(@times,s_lambda,kappa_update);
        kappa_update = sum(sum(CBIG_nanmean(sum(kappa_update,1),4),3));
        kappa_update = kappa_update./sum(sum(sum(s_lambda,1),3));
        kappa_update = invAd(setting_params.dim,kappa_update);
        kappa_update = repmat(kappa_update,1,setting_params.num_clusters);

        if (sum(kappa_update == Inf) ~= 0)
            fprintf('[WARNING] kappa is Inf !\n')
            kappa_update(kappa_update == Inf) = Params.kappa(kappa_update == Inf);
        end
        if (sum(kappa_update < setting_params.ini_concentration) ~= 0)
            kappa_update(kappa_update < setting_params.ini_concentration) = setting_params.ini_concentration;
            fprintf('[WARNING] kappa is less than the minimal value %d \n', setting_params.ini_concentration);
        end
       
        for s = 1:setting_params.num_sub
            for t = 1:setting_params.num_session
                
                % update s_t_nu
                checknu = [];
                X = data.series(:,:,s,t);
                s_lambda = Params.s_lambda(:,:,s);
                lambda_X = bsxfun(@times,kappa_update,X'*s_lambda) ...
                    + bsxfun(@times,Params.sigma,Params.s_psi(:,:,s));
                s_t_nu_update = bsxfun(@times,lambda_X,1./sqrt(sum((lambda_X).^2)));
                checknu = diag(s_t_nu_update'*Params.s_t_nu(:,:,t,s));
                checknu(isnan(checknu)) = 1;
                checknu_flag = (sum(1-checknu < setting_params.epsilon) < setting_params.num_clusters);
                Params.s_t_nu(:,:,t,s) = s_t_nu_update;

                if(checknu_flag < 1)
                    flag_nu(s,t) = 1;
                end
                
            end
        end
        if((sum(sum(flag_nu)) == setting_params.num_sub*setting_params.num_session) ...
           && (mean(abs(Params.kappa-kappa_update)./Params.kappa) < setting_params.epsilon))
            stop_m=1;
        end
        Params.kappa = kappa_update;
    end
    %% Estep
    fprintf('Estep..\n');
    
    % estimate s_lambda
    log_vmf = permute(Params.s_t_nu,[1,2,4,3]);
    log_vmf = mtimesx(data.series,log_vmf);%NxLxSxT
    log_vmf = bsxfun(@times,permute(log_vmf,[2,1,3,4]),transpose(Params.kappa));%LxNxSxT
    log_vmf(:,sum(log_vmf==0,1)==0) = bsxfun(@plus,Cdln(transpose(Params.kappa),setting_params.dim), ...
                                      log_vmf(:,sum(log_vmf==0,1)==0));%LxNxSxT
    log_vmf = CBIG_nansum(log_vmf,4);%NxLxS
    idx = sum(log_vmf == 0,1) ~= 0;
    
    log_vmf = bsxfun(@plus,permute(log_vmf,[2,1,3]),log(Params.theta));
    s_lambda = bsxfun(@minus,log_vmf,max(log_vmf,[],2));
    s_lambda = exp(s_lambda);
    Params.s_lambda = bsxfun(@times,s_lambda,1./sum(s_lambda,2));
    Params.s_lambda = permute(Params.s_lambda,[2,1,3]);
    Params.s_lambda(:,idx) = 0;
    Params.s_lambda = permute(Params.s_lambda,[2,1,3]);
    
    % estimate theta
    Params.theta = mean(Params.s_lambda,3);

    %% em stop criteria
    for s = 1:setting_params.num_sub    
        for t = 1:setting_params.num_session
            X = data.series(:,:,s,t);
            setting_params.num_verts = size(X,1);
            log_vmf = vmf_probability(X,Params.s_t_nu(:,:,t,s), Params.kappa);
            if(t == 1)
                log_lambda_prop = log_vmf;
            else
                log_lambda_prop = CBIG_nansum(cat(3,log_lambda_prop,log_vmf),3);
            end
        end
        theta_cost = Params.theta;
        log_theta_cost = log(theta_cost);
        log_theta_cost(isinf(log_theta_cost)) = log(eps.^20);

        s_lambda_cost = Params.s_lambda(:,:,s);
        log_s_lambda_cost = log(s_lambda_cost);
        log_s_lambda_cost(isinf(log_s_lambda_cost)) = log(eps.^20);

        update_cost(:,s) = sum(sum(s_lambda_cost.*log_lambda_prop)) ...
                           + sum(sum(s_lambda_cost.*log_theta_cost)) ...
                           - sum(sum(s_lambda_cost.*log_s_lambda_cost));    
    end
    sub_set=find((abs(abs(update_cost-cost)./cost) > setting_params.epsilon) == 0);
    if(length(sub_set) == setting_params.num_sub)
        stop_em = 1;
        Params.cost_em = cost;
    end
    if(iter_em > 100)
        stop_em = 1;
        Params.cost_em = update_cost;
        warning('vem can not converge');
    end
    cost = update_cost;
end
fprintf('EM..Done\n');
end


function log_vmf = vmf_probability(X,nu,kap)

% log of von Mises-Fisher distribution
% X: input data
% nu: mean direction
% kap: concentration parameter. kap is a 1xL vector

dim = size(X,2) - 1;
log_vmf = bsxfun(@plus,Cdln(kap,dim),bsxfun(@times,kap,X*nu));
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

