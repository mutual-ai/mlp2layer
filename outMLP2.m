%% 2-Layer Multilayer Perceptron Output.
% [O] = outMLP2(bias,input,w1,w2)

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

function [O] = outMLP2(bias,input,w1,w2)
%% 2-Layer Multilayer Perceptron Output.

% bias(layer) = [layer 1's bias,layer 2's bias].
% input = MLP inputs.
% w1 = neurons X weights of layer 1.
% w2 = neurons X weights of layer 2.

% Output of first layer.     
y1 = sigmoid( ( [bias(1) input]*w1' ) ,1) ;
    
% Output of second layer.
y2 = sigmoid( ( [bias(2) y1]*w2' ) ,1) ;
      
O = y2;

end
