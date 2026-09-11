function real_run_profiles(root, index)
% I/O wrapper only; real_profile_kernel is generated from pinned CBIG slices.
assets = load(fullfile(root, 'config.mat'));
stem = assets.run_stems{index};
input = load(fullfile(root, [stem '-input.mat']));
tic;
[binary, normalized, t] = real_profile_kernel(input, assets);
elapsed_seconds = toc;
runtime_version = version;
clear input assets;
temporary = fullfile(root, [stem '-reference.partial.mat']);
save('-mat-binary', temporary, 'binary', 'normalized', 't', 'elapsed_seconds', 'runtime_version');
movefile(temporary, fullfile(root, [stem '-reference.mat']));
fprintf('ORACLE_VERSION=%s\n', runtime_version);
end
