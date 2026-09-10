function profile_oracle(root)
% Equations extracted from CBIG, see ../upstream/LICENSE-CBIG.md.
input = load(fullfile(root, 'fixtures', 'profile_input.mat'));
lh_seed_ind = logical(input.lh_cortex(1:input.seed_vertices));
rh_seed_ind = logical(input.rh_cortex(1:input.seed_vertices));
s_series = [input.lh(:,lh_seed_ind) input.rh(:,rh_seed_ind)];
corr_mat1 = CBIG_corr(s_series, input.lh);
corr_mat1(isnan(corr_mat1)) = 0;
corr_mat2 = CBIG_corr(s_series, input.rh);
corr_mat2(isnan(corr_mat2)) = 0;
correlations = [corr_mat1 corr_mat2]';
threshold = '0.1';
tmp = [corr_mat1 corr_mat2];
tmp = sort(tmp(:), 'descend');
    t = tmp(round(numel(tmp) * str2num(threshold)));
    disp(['threshold: ' num2str(t)]);
        corr_mat1(corr_mat1 <  t) = 0;
        corr_mat1(corr_mat1 >= t) = 1;
            corr_mat2(corr_mat2 <  t) = 0;
            corr_mat2(corr_mat2 >= t) = 1;
binary = [corr_mat1 corr_mat2]';
series = single(input.profiles);
series(~logical(input.profile_cortex),:) = 0;
series = bsxfun(@minus,series,mean(series, 2));
                series(all(series,2)~=0,:) = bsxfun(@rdivide,series(all(series,2)~=0,:), ...
                                             sqrt(sum(series(all(series,2)~=0,:).^2,2)));

normalized = series;
series = input.avg_profile;
labels = input.labels;
% 0 label and zero correlation will be masked
medial_mask = labels ~= 0;
non_zero_corr_index = (sum(series, 2) ~= 0);
mask = medial_mask & non_zero_corr_index;

% Compute parameters based on labels
r = zeros(length(labels), max(labels));
rowindx = [1:1:length(labels)]';
r(sub2ind(size(r), rowindx(labels ~= 0), labels(labels ~= 0))) = 1;
r = r(mask, :);
x = series(mask, :);
x = bsxfun(@minus, x, mean(x, 2));
x = bsxfun(@times, x, 1./sqrt(sum(x.^2, 2)));
mtc = x' * r;
mtc = bsxfun(@times, mtc, 1./sqrt(sum((mtc).^2)));


centroids = mtc;
save('-mat7-binary',fullfile(root,'fixtures','profile_output.mat'), 'correlations', 'binary', 'normalized', 'centroids', 't');
fprintf('ORACLE_VERSION=%s\n',version);
end
