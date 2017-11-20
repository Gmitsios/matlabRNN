function [X, out] = runRNN(net, param, IN, REC)
% Forward running of RNN


OUP = param.output_N;
NOL = param.num_layers;
NOD = param.num_nodes;

Win = net.win;
W = net.w;
Wn = net.wn;
Wout = net.wout;

X = zeros(NOL,NOD);


for j = (1:NOD)
     X(1,j) = tanh1(Win(j,:)*[REC(1,:), IN]');
end


for i = (2:NOL-1)
    for j = (1:NOD)
        X(i,j) = tanh1(W((i-2)*NOD+j,:)*[REC(i,:), X(i-1,:)]');
    end
end


for j = (1:NOD)
    X(NOL,j) = tanh1(Wn(j,:)*X(NOL-1,:)');
end


for j = (1:OUP)
    out = tanh1(Wout(j,:)*X(NOL,:)');
end