%% Learning XOR logical operation

% Configuration
neurons = [4 1];
a = 1e-3;
bias = [-1 -1];
x = [0 0;0 1;1 0; 1 1];
yref = [0 ;1 ;1 ;0];
lr = 0.7;
error = 1e-20;
maxIt = 1e4;

% Processing
[W1,W2,E] = trainingMLP2(neurons,a,bias,x,yref,lr,error,maxIt);

% Display Infos
disp('Error');
disp(E);
disp('XOR')
disp('Input [0 0]');
outMLP2(bias,[0 0],W1,W2)
disp('Input [0 1]');
outMLP2(bias,[0 1],W1,W2)
disp('Input [1 0]');
outMLP2(bias,[1 0],W1,W2)
disp('Input [1 1]');
outMLP2(bias,[1 1],W1,W2)

% Plot MSE
semilogx(mse_hist)
ylabel('MSE');
xlabel('Iteration');
title('Xor Example','FontSize',14);
