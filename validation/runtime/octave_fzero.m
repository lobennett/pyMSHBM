function [root, value, flag] = octave_fzero(fun, x0)
% Runtime compatibility only. CBIG invAd returns outu, never this root.
try
    [root, value, flag] = fzero(fun, x0);
catch error_info
    if isempty(strfind(error_info.message, 'not a valid initial bracketing'))
        rethrow(error_info);
    end
    root = x0;
    value = fun(x0);
    flag = -1;
end
end
