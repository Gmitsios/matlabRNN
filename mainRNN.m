% RNN main function


%Network parameters
params.input_N = 1;             % Number of inputs
params.output_N = 1;            % Number of outputs
params.num_layers = 2;          % Number of layers (>=2)
params.num_nodes = 4;           % Number of nodes/layer

% Initialize Global variables
global EPOCHS
global SUB_LEN
global SEQ

EPOCHS = 2000;                  % Number of epochs
SUB_LEN = 100;                  % Length of subset
SEQ = 1000;                     % Length of training sequence


% Initialize weights matrices
[ net ] = weights_init(params);


% Train network
[t_net, Xlast] = trainRNN(net, params);


%Test trained network
test_len = 1000;
[Test_Data, Data_True]  = datagen_rank3e(test_len, test_len, 1);

RNN_OUT = zeros(test_len,1);
true = 0;
for i=1:test_len
   [X, out] = runRNN(t_net, params, Test_Data(i,:), Xlast); 
   RNN_OUT(i) = out(1);
       
end

rmse = sqrt(mean((RNN_OUT(1:end) - Data_True(1:end)).^2));
fprintf('** Test data MSE: %f **\n', rmse);


%Evaluation
[s1, s2] = RNN_evaluation(Test_Data, Data_True, RNN_OUT);


%Plot
plot_result( s1, s2 );
