function [ netW ] = weights_init( param )

IP = param.input_N;
OP = param.output_N;
NOL = param.num_layers;
NOD = param.num_nodes;


Win = zeros(NOD, IP+NOD);
for i=1:NOD
    for j=1:IP+NOD
        Win(i,j) = rand ./ 2  - 0.25;
    end
end
netW.win = Win;

W = zeros((NOL-2)*NOD, 2*NOD);
for i=1:(NOL-2)*NOD
    for j=1:2*NOD
        W(i,j) = rand ./ 2  - 0.25;
    end
end
netW.w = W;

Wn = zeros(NOD, NOD);
for i=1:NOD
    for j=1:NOD
        Wn(i,j) = rand ./ 2  - 0.25;
    end
end
netW.wn = Wn;

Wout = zeros(OP, NOD);
for i=1:OP
    for j=1:NOD
        Wout(i,j) = rand ./ 2  - 0.25;
    end
end
netW.wout = Wout;

end
