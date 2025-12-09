%Classification Neural Network by Sacha Muller
%AERO 40041 Data-Driven Methods CW2 
%Group 61

%Pseudocode 

%1. Data Import
%2. Structure determination (number of layers, neurons)
%3. Initial weights 
%4. Input activation functions
%5. Implement forward propagation 
%6. Implement cost function
%7. Implement backwards propagation
%8. Implement training function (running forward and back propagation)

%% Data Import
%import csv
clear
trainingdata = readmatrix('classification_training.csv');

%cut out NaN, don't run this codeblock without re-import csv or you will
%delete data
trainingdata(1,:) = [];
feature1 = trainingdata(:,[1 2]);
feature2 = trainingdata(:, [1 3]);
trainingclass= trainingdata(:, [1 4]);
window1 = [min(feature1(:,2)), max(feature1(:,2))];
window2 = [min(feature2(:,2)), max(feature2(:,2))];

%% Structure and Initial Weights
%constants -> change as necessary%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
inputfeatures = 2;
neuronperlayer = 10;
outputneurons = 1;
layernumber = 1 ;%code can be generalized to arbitrary layer number
runhyperparamsweep = 1; %set to 0 to avoid running parameter sweep
rng(12345)

%creating weights of 1st layer
w{1} = 0.5 * rand(neuronperlayer,inputfeatures) - 0.25;

%Weights and biases of additional layers
%I know I should pre-allocate matrices for efficiency but they're small
%so its ok
for i=1:layernumber
    b{i} = zeros(neuronperlayer,1);
end

if layernumber > 1

    for i = 2:layernumber
    w{i} = 0.5 * rand(neuronperlayer,neuronperlayer) - 0.25;
    end
end

%Weights and biases of final layer 

w{layernumber + 1} = 0.5 * rand(outputneurons,neuronperlayer) - 0.25;
b{layernumber + 1} = zeros(outputneurons, 1) ;

%% Training Loop  

%TRAINING LOOP CONSTANTS%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
alpha = 0.06; %learning rate;
epsilon = 1e-15; %avoid division by 0;
epochs = 100;

for i=1:epochs
%shuffle data so that every epoch has a different dataset
    shuffler = randperm(length(trainingdata));

for j=1:length(trainingdata) 
%feedforward through dataset one sample at a time
samplex = [feature1(shuffler(j),2); feature2(shuffler(j),2)];
sampley = [trainingclass(shuffler(j),2)];
  
[n, a] = forward(samplex, w, b);

%feedback

[w, b] = backward(samplex, sampley, n, a, w, b, alpha, layernumber) ; 

%evaluate cost
cost(j) = -(sampley * log(a{end} + epsilon) + (1 - sampley) * log(1 - a{end} + epsilon));

%cost check every few hundred iterations 

if mod(j,100) == 0
    cumulcost = sum(cost)/j;
    disp(['the cost at iteration ' num2str(j) ' is ' num2str(cumulcost)])
end

end
end

%% --- Validation and Plotting ---

% 1. Import Validation Data
    validation_data = readmatrix('classification_validation.csv');
    validation_data(1,:) = []; % Remove NaN

val_x1 = validation_data(:, 2); 
val_x2 = validation_data(:, 3);
val_y  = validation_data(:, 4);

% 2. Generate Grid 
resolution = 0.1; % adjust for smooth contours
x_min = min([feature1(:,2); val_x1]) - 0.2;
x_max = max([feature1(:,2); val_x1]) + 0.2;
y_min = min([feature2(:,2); val_x2]) - 0.2;
y_max = max([feature2(:,2); val_x2]) + 0.2;

[xx, yy] = meshgrid(x_min:resolution:x_max, y_min:resolution:y_max);
grid_points = [xx(:), yy(:)];

% 3. Predict on Grid
Z = zeros(size(grid_points, 1), 1);

for k = 1:size(grid_points, 1)
    sample_grid = grid_points(k, :)'; 
    
    [~, a_out] = forward(sample_grid, w, b);
    
    Z(k) = a_out{end}; 
end

% Reshape Z back to the grid shape for contour plotting
Z = reshape(Z, size(xx));

% 4. Create the Contour Plot 
figure('Name', 'Decision Boundary', 'Color', 'w');
hold on;

% Draw the probability map
contourf(xx, yy, Z, 50, 'LineColor', 'none'); 
colormap(jet); 
colorbar;
clim([0 1]);

% Draw decision boundary (Z = 0.5)
[C, h] = contour(xx, yy, Z, [0.5 0.5], 'k', 'LineWidth', 3);
clabel(C, h, 'FontSize', 12, 'Color', 'white');

%  Plot the Validation Data
% Plot Class 0 points
idx0 = (val_y == 0);
scatter(val_x1(idx0), val_x2(idx0), 60, 'o', 'filled', ...
    'MarkerFaceColor', 'cyan', 'MarkerEdgeColor', 'k', 'DisplayName', 'Class 0');

% Plot Class 1 points
idx1 = (val_y == 1);
scatter(val_x1(idx1), val_x2(idx1), 60, 's', 'filled', ...
    'MarkerFaceColor', 'magenta', 'MarkerEdgeColor', 'k', 'DisplayName', 'Class 1');

% Formatting
title('Validation Data & Decision Boundary');
xlabel('Feature 1');
ylabel('Feature 2');
legend('Location', 'best');
grid on;
hold off;


%% --- Hyperparameter Sweep ---
%constants -> change as necessary
if runhyperparamsweep == 1

neurontrain = 10:10:250;
inputfeatures = 2;
layernumber = 1; %code can be generalized to arbitrary layer number
rng(12345)
neuronindex = 1;
alphatrain = 0.01:0.01:0.25; %learning rate;
epsilon = 1e-15; %avoid division by 0;
epochtrain = 25;
val_X = [val_x1 val_x2];
    total_val = length(val_X);
for neuronperlayer = neurontrain
  
alphaindex = 1;
for alpha = alphatrain

%creating weights of 1st layer
wtrain{1} = 0.5 * rand(neuronperlayer,inputfeatures) - 0.25;

%initialise Weights and biases of additional layers

for i=1:layernumber
    btrain{i} = zeros(neuronperlayer,1);
end

if layernumber > 1

    for i = 2:layernumber
    wtrain{i} = 0.5 * rand(neuronperlayer,neuronperlayer) - 0.25;
    end
end

%initialise Weights and biases of final layer 

wtrain{layernumber + 1} = 0.5 * rand(outputneurons,neuronperlayer) - 0.25;
btrain{layernumber + 1} = zeros(outputneurons, 1) ;




    
for i=1:epochtrain
%shuffle data so that every epoch has a different dataset
    shuffler = randperm(length(trainingdata));
    clear cost cumulcost 

for j=1:length(trainingdata) 
%feedforward through dataset one sample at a time
samplex = [feature1(shuffler(j),2); feature2(shuffler(j),2)];
sampley = [trainingclass(shuffler(j),2)];

[n, a] = forward(samplex, wtrain, btrain);

[wtrain, btrain] = backward(samplex, sampley, n, a, wtrain, btrain, alpha, layernumber) ; 

%evaluate cost
cost(j) = -(sampley * log(a{end} + epsilon) + (1 - sampley) * log(1 - a{end} + epsilon));
%cost check every iteration

if mod(j,100) == 0
    cumulcost = sum(cost)/j;
    disp(['the cost at iteration ' num2str(j) ' is ' num2str(cumulcost)])
end

end
end
finalcost(alphaindex,neuronindex) = cumulcost;

%check accuracy against validation data
% put a forward pass here on the validation data
    correct_count = 0;
    
    for v = 1:total_val
                vx = val_X(v,:)';
                vy = val_y(v);
                
                [~, va_out] = forward(vx, wtrain, btrain);
                prediction = va_out{end} >= 0.5;
                
                if prediction == vy
                    correct_count = correct_count + 1;
                end
    end
            
            accuracy(alphaindex, neuronindex) = (correct_count / total_val) * 100;
            
            alphaindex=alphaindex+1;

end
neuronindex = neuronindex + 1;
end

mincost = min(finalcost, [], "all");
[mincostrow, mincostcol] = find(finalcost == mincost);

maxacc = max(accuracy, [], "all");
[maxaccrow, maxacccol] = find(accuracy == maxacc);

alphacost = alphatrain(mincostrow);
neuroncost = neurontrain(mincostcol);

alphaacc = alphatrain(maxaccrow);
neuronacc = neurontrain(maxacccol);

disp(['The minimal cost learning rate is ' num2str(min(alphacost)) ' and the neuron number is ' ...
    num2str(min(neuroncost))])

disp(['The maximum accuracy learning rate is ' num2str(min(alphaacc)) ' and the neuron number is ' ...
    num2str(min(neuronacc))])

%at the end of every alpha loop, check the training data

end


%% Backwards Func 

function [w, b] = backward(x, y, n, a, w, b, alpha, layernumber)

% 1. dLoss/doutput -> BCE with sigmoid means it's guess - true
dl_dout{layernumber + 1} = a{end} - y;

%2. gradients of activation functions
%gradient of tanh, use n1 because tanh


%gradient of sigmoid, set to 1 because our derivative of loss / output
%includes the sigmoid
fprime{layernumber + 1} = 1; %a{end}*(1-a{end}); %only for last layer

for j = 1:layernumber

fprime{j} = tanhgrad(n{j}); %same for all hidden layers

end


%3. Derivative of loss w.r.t hidden layers
for j = layernumber+1:-1:1



if j-1 > 0
dl_dout{j-1} = w{j}' * (dl_dout{j} .* fprime{j});
end


%5. calculate dl/db (or what's inside the brackets)
dl_db{j} = dl_dout{j}.*fprime{j}; %using hadamar

%5. calculate dl_dw
if j == 1
    dl_dw{j} = dl_db{j}*x';

else
    dl_dw{j} = dl_db{j}*a{j-1}';
end
end

for j=1:layernumber+1
w{j} = w{j} - alpha*dl_dw{j};
b{j} = b{j}- alpha*dl_db{j};

end

end

%% Forward Function 

function [n, a] = forward(x, w, b)

layernumber = length(w);

for l = 1:layernumber

    if l == 1
        n{l} = w{l} * x + b{l};
        a{l} = tanh(n{l});
    

    elseif l == layernumber
        n{l} = w{l} * a{l-1} + b{l};
        a{l} = mysig(n{l});
    
    else 
        n{l} = w{l} * a{l-1} + b{l};
        a{l} = tanh(n{l});
    end

end
end




%% Math Funcs

function grad = tanhgrad(x)

grad = 1-tanh(x).^2;

end

%sigmoid trick
function sigout = mysig(z)
    if( z>=0 )
        sigout = 1 / (1 + exp(-z));
    else
         sigout = exp(z) / (1 + exp(z));
    end
end



