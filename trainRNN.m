function [trained_net, Xlast] = trainRNN(net, param)


INP = param.input_N;
OUP = param.output_N;
NOL = param.num_layers;
NOD = param.num_nodes;

% Set Global variables
global EPOCHS
global SUB_LEN
global SEQ

num_Epoch = EPOCHS;             % Number of epochs
num_sub = SUB_LEN;              % Length of subset
sig_seq = SEQ;                  % Length of training sequence


% Get weights from struct
Win = net.win;
W = net.w;
Wn = net.wn;
Wout = net.wout;

% Generate cross evaluation data
[cv_INP, cv_TRG]  = datagen_rank3e(sig_seq, sig_seq, 1);
cv_OUT = zeros(sig_seq,1);

% Initialize some variables for training

% Starting point of subset in data sequence
start = ceil((sig_seq-num_sub-(INP+OUP)+2)*rand(1,num_Epoch));
% Anneal R from 100 to 5
R = annealing(100,5,num_Epoch);
% Anneal Q from 1E-2 to 1E-6
Q = annealing(1E-2,1E-6,num_Epoch);
% Learning rate
learning_rate = annealing(1,1E-4,num_Epoch);

now = 1;
min_mse = 1;
updated=0;

% Training
for t = (1:num_Epoch)
    
    % Get training data
    [DInput, Dtarget]  = datagen_rank3e(sig_seq, num_sub, start(t));
    [INPsize, INPnum] = size(DInput');
    [OUTsize, OUTnum] = size(Dtarget');
    
    % Initalize output and X matrix
    % X0, X1, X2, X3 - output of each layer at different time steps
    out = zeros(INPnum,1);  
    X0 = zeros(NOL,NOD);
    
    %2 first runs of RNN (initialization)
    [X, out(1)] = runRNN(net, param, DInput(1,:), X0);
    X1 = X;

    [X, out(2)] = runRNN(net, param, DInput(2,:), X1);
    X2 = X;
        
    
    % Ricatti equation initialization
    for m = (1:NOL*NOD+OUP)
        if (m<NOD+1)
            K(m).value = 0.01^(-1)*eye(NOD+INP);
        elseif (m<2*NOD+1)            
            if (NOL<3)
                K(m).value = 0.01^(-1)*eye(NOD);
            else
                K(m).value = 0.01^(-1)*eye(2*NOD);
            end
        elseif (m<(NOL-1)*NOD+1)
            K(m).value = 0.01^(-1)*eye(2*NOD);
        elseif (m<NOL*NOD+1)
            K(m).value = 0.01^(-1)*eye(NOD);
        else
            K(m).value = 0.01^(-1)*eye(NOD);
        end
    end
        
    W0 = zeros(NOD,NOD+INP);
    
    % Remaining runs of RNN
    for k = (3:INPnum)
        
        temp = 0;
        
        % Forward running of network
        [X, out(k)] = runRNN(net, param, DInput(k,:), X2);
        X3 = X;
        
        % Get updated weights from struct
        Win = net.win;
        W = net.w;
        Wn = net.wn;
        Wout = net.wout;
        
        %Backpropagation of error and jacobian matrix 'C' calculation    
        %
        % -> @ output neuron
        for j = (NOL*NOD+1:NOL*NOD+OUP)
            C(j).value = X3(NOL,:);
        end
        
        
        % -> @ non recurrent layer
        D = Wout*diag(d_tanh1(Wn*X3(NOL-1,:)'));
        D1 = D'*X3(NOL-1,:);
        for j = (1:NOD)
            C((NOL-1)*NOD+j).value = D1(j,:);
        end
        
        
        if NOL>2
        if NOL>3
            
        % -> @ last recurrent layer
        D = D*Wn*diag(d_tanh1(W((NOL-3)*NOD+1:(NOL-2)*NOD,:)*[X2(NOL-1,:) X3(NOL-1,:)]'));
        D1 = D'*[X2(NOL-1,:) X3(NOL-1,:)];
        D2 = D1 + (D*W((NOL-3)*NOD+1:(NOL-2)*NOD,1:NOD)*diag(d_tanh1(W((NOL-4)*NOD+1:(NOL-3)*NOD,:)* ...
            [X1(NOL-1,:) X2(NOL-1,:)]')))' * [X1(NOL-1,:) X2(NOL-1,:)];
        for j = (1:NOD)
            C((NOL-2)*NOD+j).value = D2(j,:);
        end
        
        
        % -> @ previous recurrent layers
        for i = (NOL-3:-1:2)
            D = D*W(i*NOD+1:(i+1)*NOD,1:NOD)*diag(d_tanh1(W((i-1)*NOD+1:i*NOD,:)*[X2(i,:) X3(i,:)]'));
            D1 = D'*[X2(i,:) X3(i,:)];
            D2 = D1 + (D*W((i-1)*NOD+1:i*NOD,1:NOD)*diag(d_tanh1(W((i-2)*NOD+1:(i-1)*NOD,:)* ...
            [X1(i,:) X2(i,:)]')))' * [X1(i,:) X2(i,:)];
            for j = (1:NOD)
               C((NOL-1-i)*NOD+j).value = D2(j,:);
            end
          
        end
        
        end
        
        % -> @ 2nd recurrent layer
        if (NOL>3)
            D = D*W(NOD+1:2*NOD,1:NOD)*diag(d_tanh1(W(1:NOD,:)*[X2(2,:) X3(2,:)]'));
        else
            D = D*Wn*diag(d_tanh1(W(1:NOD,:)*[X2(2,:) X3(2,:)]'));
        end
        D1 = D'*[X2(2,:) X3(2,:)];
        if INP>NOD
            D2 = D1 + (D*W(1:NOD,1:NOD)*diag(d_tanh1(Win(:,1:2*NOD)*[X1(2,:) X2(2,:)]')))' * [X1(2,:) X2(2,:)];
            
        else
            Wtemp = zeros(NOD,2*NOD);
            Wtemp(:,1:NOD+INP) = Win;
            D2 = D1 + (D*W(1:NOD,1:NOD)*diag(d_tanh1(Wtemp(:,1:2*NOD)*[X1(2,:) X2(2,:)]')))' * [X1(2,:) X2(2,:)];
        end
        
        for j = (1:NOD)
            C(NOD+j).value = D2(j,:);
        end
        
        end
        
        % ---
        
        % -> @ 1st recurrent layer
        if (NOL<3)
            D = D*Wn(1:NOD,1:NOD)*diag(d_tanh1(Win*[X2(1,:) DInput(k,:)]'));
        else
            D = D*W(1:NOD,1:NOD)*diag(d_tanh1(Win*[X2(1,:) DInput(k,:)]'));
        end
        D1 = D'*[X2(1,:) DInput(k,:)];
        D2 = D1 + (D*Win(:,1:NOD)*diag(d_tanh1(W0*[X1(1,:) DInput(k-1,:)]')))' * [X1(1,:) DInput(k-1,:)];
        for j = (1:NOD)
            C(j).value = D2(j,:);
        end
        
        Winput = Win;
        
        
        %Decoupled EKF
        alpha = Dtarget(k) - out(k);    % Innovation of output
        
        for m = (1:NOL*NOD+OUP)
            temp = C(m).value*K(m).value*C(m).value' + temp;
        end
        
        %Inverse of temp+R(t)
        temp2 = temp+R(t);
        [U, S, V]= svd(temp2);
        s= diag(S); p= sum(s> 1e-9);
        Gamma = (U(:, 1: p)* diag(1./ s(1: p))* V(:, 1: p)')';
        
        for m = (1:NOL*NOD+OUP)
            G(m).value = K(m).value*C(m).value'*Gamma;
            
            %update weights if innovation > threshold
            if abs(alpha) > 1E-3  %|| wrong<0
                updated = updated+1;
                if (m<NOD+1)
                    Win(m,:) = Win(m,:) + learning_rate(t)*(G(m).value*alpha)';
                    %Win(m,:) = Win(m,:) + 0.2*(G(m).value*alpha)';
                end
                if (m>NOD && m<(NOL-1)*NOD+1)
                    W(m-NOD,:) = W(m-NOD,:) + learning_rate(t)*(G(m).value*alpha)';
                    %W(m-NOD,:) = W(m-NOD,:) + 0.2*(G(m).value*alpha)';
                end
                if (m>(NOL-1)*NOD && m<NOL*NOD+1)
                    Wn(m-(NOL-1)*NOD,:) = Wn(m-(NOL-1)*NOD,:) + learning_rate(t)*(G(m).value*alpha)';
                    %Wn(m-(NOL-1)*NOD,:) = Wn(m-(NOL-1)*NOD,:) + 0.2*(G(m).value*alpha)';
                end
                if (m>NOL*NOD)
                    Wout(m-NOL*NOD,:) = Wout(m-NOL*NOD,:) + learning_rate(t)*(G(m).value*alpha)';
                    %Wout(m-NOL*NOD,:) = Wout(m-NOL*NOD,:) + 0.2*(G(m).value*alpha)';
                end              
            end
            
            % Re-calculte ricatti equation
            K(m).value = K(m).value - G(m).value*C(m).value*K(m).value + Q(t);
            
            net.win = Win;
            net.w = W;
            net.wn = Wn;
            net.wout = Wout;
            
            %for i=1:sig_seq
            %    [X, tmp_out] = runRNN(net, param, cv_INP(i,:), X3); 
            %    cv_OUT(i) = tmp_out(1);       
            %end
            %cv_mse = sqrt(mean((tmp_out(1:end) - cv_TRG(1:end)).^2));
            %if cv_mse<0.0001, break; end;
            
        end
        
        %End of Decoupled EKF
        
        % Update recurrent states for next run
        X1 = X2;
        X2 = X3;
        % Replace 1st layer weights
        W0 = Winput;       
                
    end
    
    % Calculate MSE
    mse(t) = sqrt(mean((out(1:end) - Dtarget(1:end)).^2));
    dis(t)= sum(abs(out(1:end) - Dtarget(1:end)));
    
    
    % Calculate cross evaluation MSE
    for i=1:sig_seq
        [X, tmp_out] = runRNN(net, param, cv_INP(i,:), X3); 
        cv_OUT(i) = tmp_out(1);       
    end
    cv_mse = mean((cv_OUT(1:end) - cv_TRG(1:end)).^2);   
    
    if cv_mse<min_mse
        now = t;
        min_mse = cv_mse;
        X_min = X3;
        min_Win = Win;
        min_W = W;
        min_Wn = Wn;
        min_Wout = Wout;
    end
    
    fprintf('Epoch: %d, Bunch MSE: %f, Total Distance: %f, Cross MSE: %f\n', t, mse(t), dis(t), cv_mse);
    %if mse(t)<0.001 break; end;
    if cv_mse<0.00001, break; end;
    
end

fprintf('\nWeights Updated %d Times\n',updated/(NOL*NOD+OUP));
fprintf('Best net at epoch %d with cross mse of %f\n\n\n',now, min_mse);

% Return variables
Xlast = X_min;
trained_net.win = min_Win;
trained_net.w = min_W;
trained_net.wn = min_Wn;
trained_net.wout = min_Wout;


end