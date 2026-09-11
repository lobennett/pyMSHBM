function real_run_kernel(root)
% I/O wrapper: full single tensor and independently constructed raw centroids.
config = load(fullfile(root, 'config.mat'));
shape = double(config.shape);
data.series = NaN(shape, 'single');
for index = 1:numel(config.run_stems)
    input = load(fullfile(root, [config.run_stems{index} '-reference.mat']), 'normalized');
    assert(isa(input.normalized, 'single'));
    data.series(:,:,config.subject_indices(index),config.session_indices(index)) = input.normalized;
    clear input;
end
input = load(fullfile(root, 'reference-centroids.mat'));
g_mu = input.g_mu;
clear input;
tic;
Params = cbig_kernel(data, g_mu, double(config.max_iter));
elapsed_seconds = toc;
runtime_version = version;
temporary = fullfile(root, 'reference-params.partial.mat');
save('-mat-binary', temporary, 'Params', 'elapsed_seconds', 'runtime_version');
movefile(temporary, fullfile(root, 'reference-params.mat'));
fprintf('ORACLE_VERSION=%s\n', runtime_version);
end
