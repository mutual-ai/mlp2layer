%% Learning XOR and AND logical operation

% Configuration
neurons = [5 2];
a = 0.001;
bias = [-1 -1];
x = [0 0;0 1;1 0; 1 1];
yref = [0 0;1 0;1 0;0 1];
lr = 0.7;
error = 10^-20;
maxIt = 100000;

% Processing
[W1,W2,E] = trainingMLP2(neurons,a,bias,x,yref,lr,error,maxIt);

% Display Infos
disp('Error');
disp(E);
disp('XOR AND')
disp('Input [0 0]');
outMLP2(bias,[0 0],W1,W2)
disp('Input [0 1]');
outMLP2(bias,[0 1],W1,W2)
disp('Input [1 0]');
outMLP2(bias,[1 0],W1,W2)
disp('Input [1 1]');
outMLP2(bias,[1 1],W1,W2)

% Plot MSE
plot(mse_hist)
