function result = mtimesx(a, b)
% Infrastructure substitute for the two non-transposed page products used by
% CBIG's group-prior kernel. No estimator equations are implemented here.
assert(size(a,2) == size(b,1));
assert(size(a,3) == size(b,3));
assert(size(a,4) == size(b,4));
% Pinned mtimesx.c lines 755-756/783-784 convert the double matrix to
% single before FloatTimesFloat when either non-scalar matrix is single.
if isa(a, 'single') || isa(b, 'single')
    a = single(a);
    b = single(b);
end
result = zeros(size(a,1), size(b,2), size(a,3), size(a,4), class(a));
for subject = 1:size(a,3)
    for session = 1:size(a,4)
        result(:,:,subject,session) = a(:,:,subject,session) * b(:,:,subject,session);
    end
end
end
