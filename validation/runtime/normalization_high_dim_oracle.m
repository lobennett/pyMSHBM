function normalization_high_dim_oracle(root)
% Infrastructure wrapper; normalization body copied verbatim from pinned CBIG.
input = load(fullfile(root, 'fixtures', 'normalization_high_dim_input.mat'));
normalized = normalize_only(input.profiles, input.cortex);
single_normalized = zeros(size(normalized), 'single');
for row = 1:size(input.profiles, 1)
    single_normalized(row,:) = normalize_only(input.profiles(row,:), input.cortex(row));
end
runtime_version = version;
runtime_computer = computer;
runtime_blas = __octave_config_info__('BLAS_LIBS');
save('-mat7-binary', fullfile(root, 'fixtures', 'normalization_high_dim_output.mat'), ...
     'normalized', 'single_normalized', 'runtime_version', 'runtime_computer', 'runtime_blas');
fprintf('ORACLE_VERSION=%s\nCOMPUTER=%s\nBLAS=%s\n', runtime_version, runtime_computer, runtime_blas);
end

function series = normalize_only(profiles, cortex)
% Cast and supplied cortex mask replace MRI and mesh I/O only.
series = single(profiles);
series(~logical(cortex),:) = 0;
series = bsxfun(@minus,series,mean(series, 2));
                series(all(series,2)~=0,:) = bsxfun(@rdivide,series(all(series,2)~=0,:), ...
                                             sqrt(sum(series(all(series,2)~=0,:).^2,2)));

end
