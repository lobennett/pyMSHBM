function s2_estep_witness(root)
input=load(fullfile(root,'input.mat')); data.series=input.series;
setting_params.dim=size(data.series,2)-1;
names={'reference_state','python_state','python_nu_reference_theta','reference_nu_python_theta'};
output=struct();
for index=1:numel(names)
    name=names{index}; Params=input.(name);
    raw_dot=mtimesx(data.series,permute(Params.s_t_nu,[1,2,4,3]));

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


    output.(name).posterior=Params.s_lambda;
    output.(name).theta=Params.theta;
    output.(name).raw_dot=raw_dot;
end
runtime_version=version;
save('-mat-binary',fullfile(root,'octave.mat'),'output','runtime_version');
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

