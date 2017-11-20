function [ s1, s2 ] = RNN_evaluation( Test_Data, Data_True, RNN_OUT )

%eval_mat = [ Data_True RNN_OUT ];

mat1 = [Test_Data Data_True];
mat2 = [Test_Data RNN_OUT];
s1 = sortrows(mat1);
s2 = sortrows(mat2);

end

