function [ I_data, T_data ] = datagen_rank3e( num_points, len_sub, start )

IN = annealing(1/num_points,1,num_points)';
OUT = ((6*IN-2).^3-2*(6*IN-2).^2-4*(6*IN-2)+12)/30;

mat = [IN OUT];
[n m] = size(mat);
A = mat(randperm(n), :);
INP = A(:,1);
OUTP = A(:,2);

if (start+len_sub-1<=num_points)
    I_data = INP(start:start+len_sub-1,:);
    T_data = OUTP(start:start+len_sub-1);
else
    s = num_points-start;
    n = len_sub - s;
    I_data = [INP(start:end,:); IN(1:n-1,:)];
    T_data = [OUTP(start:end); OUT(1:n-1)];
end



end