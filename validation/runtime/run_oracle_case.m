function run_oracle_case(root, name)
% File/log adapter, with no numerical changes to the called source kernel.
if strcmp(name, 'complete')
    suffix = '';
else
    suffix = ['_' name];
end
addpath(fullfile(root, 'runtime'));
addpath(fullfile(root, 'upstream'));
input = load(fullfile(root, 'fixtures', ['synthetic_input' suffix '.mat']));
data.series = input.series;
tic;
captured = evalc('Params=cbig_kernel(data,input.g_mu,5);');
elapsed_seconds = toc;
save('-mat7-binary', fullfile(root, 'fixtures', ['synthetic_output' suffix '.mat']), 'Params', 'elapsed_seconds');
fid = fopen(fullfile(root, 'fixtures', ['octave' suffix '.log']), 'w');
fputs(fid, captured);
fprintf(fid, 'ORACLE_VERSION=%s\n', version);
fclose(fid);
fprintf('%s: iteration %d, %.3fs\n', name, Params.iter_inter, elapsed_seconds);
end
