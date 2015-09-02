%% 2-Layer Multilayer Perceptron Training.
% [W1,W2,E] = trainingMLP2(neurons,a,bias,x,yref,lr,error,maxIt)

% % Copyright (c) 2015, Augusto Damasceno.
% % All rights reserved.
% % Redistribution and use in source and binary forms, with or without modification,
% % are permitted provided that the following conditions are met:
% %   1. Redistributions of source code must retain the above copyright notice,
% %      this list of conditions and the following disclaimer.
% %   2. Redistributions in binary form must reproduce the above copyright notice,
% %      this list of conditions and the following disclaimer in the documentation
% %      and/or other materials provided with the distribution.
% % THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
% % ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
% % WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
% % IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
% % INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
% % BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
% % OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
% % WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
% % ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
% % OF SUCH DAMAGE.

function [W1,W2,E] = trainingMLP2(neurons,a,bias,x,yref,lr,error,maxIt)
%% 2-Layer Multilayer Perceptron Training.

% neurons = [layer 1's number of neurons,layer 2's number of neurons].
% a = momentum constant.
% bias(layer) = [layer 1's bias,layer 2's bias].
% x = examples X inputs.
% yref = examples X desired outputs.
% lr = learning-rate.
% error = acceptable error.
% maxIt = maximum iteration.

%% Configuration and architecture.

[examples,nInputs] = size(x);
[~,nOutputs] = size(yref);

w1 = rand(neurons(1),nInputs+1);
w1Past = zeros(neurons(1),nInputs+1);
w1tmp = zeros(neurons(1),nInputs+1);

w2 = rand(neurons(2),neurons(1)+1);
w2Past = zeros(neurons(2),neurons(1)+1);
w2tmp = zeros(neurons(2),neurons(1)+1);


%% Training. 

ex = 0;
counter = 0;

progress = 0;
displayStep = ceil(0.05*maxIt);

% Layers 1's outputs.
y1 = zeros(1,neurons(1));
% Layers 2's outputs.
y2 = zeros(1,neurons(2));
% Layer 2's gradients
Gs = zeros(1,neurons(2));

% Memory of MLP Outputs.
ys = zeros(examples,nOutputs);

% Mean Square Error.
mse = +Inf;

% Memory of Mean Square Error.
mse_hist = zeros(ceil(maxIt/examples),1);
mse_counter = 1;

% Random order of inputs - desired outputs.
xidx = randperm(examples);

while (mse > error && counter < maxIt)

    % Iteration Number.
    counter = counter + 1;
    
    % Number of training example.
    ex = mod(ex,examples) + 1;
    
    % Propagation.
    
    % Output of first layer.     
    y1 = sigmoid( ( [bias(1) x(xidx(ex),:)]*w1' ) ,1) ;
    
    % Output of second layer.
    y2 = sigmoid( ( [bias(2) y1]*w2' ) ,1) ;
    
    % Backpropagation.
    
    % Neuron Update = learning-rate*G*y(layer-1)

    % Error = yref - y
    E = yref(xidx(ex),:) - y2;
    % Derivative of the activation function = df
    df = y2.*(1-y2);
    % Local gradient. Last layer = Error * df.
    Gs = E .* df;
    
    % Update second layer.
    w2tmp = w2;
    w2 = w2 + a*w2Past + lr*Gs'*[bias(2) y1];
    
    % SUM GsW2
    WtGs = w2'*Gs';
    % Derivative of the activation function = df2
    df2 = (y1.*(1-y1));
    % Local gradient. Hidden layers: df * sum of (next layer G * next layer weights)
    Gs2 = df2.*WtGs(2:end,:)';
    
    % Update first layer.
    w1tmp = w1;
    w1 = w1 + a*w1Past + lr*Gs2'*[bias(1) x(xidx(ex),:)];
    
    % Save past weights.
    w1Past = w1tmp;
    w2Past = w2tmp;
    
    % Mean Square Error.
    ys(xidx(ex),:) = y2;
    if ex == examples
        mse = mseb(ys,yref);
        % Change order of inputs - desired outputs.
        xidx = randperm(examples);
        % Save history of MSE.
        mse_hist(mse_counter) = mse;
        mse_counter = mse_counter + 1;
    end
    
    % Display the progress.
    if mod(progress,displayStep) == 0
    	message = sprintf('%.2f%% of maximum iteration.\n',(counter*100)/maxIt);
    	disp(message);
        message = sprintf('MSE: %.4e\n',mse);
        disp(message);
    	drawnow();
    end
    progress = progress+1;
end

fprintf('\nTotal Iterations: %d\nError: %.4f\n',counter,mse);

assignin('base','mse_hist',mse_hist);

W1 = w1;
W2 = w2;
E = mse;

end
