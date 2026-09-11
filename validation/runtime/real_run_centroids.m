function real_run_centroids(root)
% Average only independent upstream binary profiles, then execute source slice.
config = load(fullfile(root, 'config.mat'));
shape = double(config.shape);
average = zeros(shape(1), shape(2));
tic;
for index = 1:numel(config.run_stems)
    input = load(fullfile(root, [config.run_stems{index} '-reference.mat']), 'binary');
    average = average + double(input.binary);
    clear input;
end
average = average ./ numel(config.run_stems);
g_mu = real_centroid_kernel(average, double(config.labels));
elapsed_seconds = toc;
runtime_version = version;
temporary = fullfile(root, 'reference-centroids.partial.mat');
save('-mat-binary', temporary, 'g_mu', 'elapsed_seconds', 'runtime_version');
movefile(temporary, fullfile(root, 'reference-centroids.mat'));
fprintf('ORACLE_VERSION=%s\n', runtime_version);
end
