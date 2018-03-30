clear all;
gpuDevice(2); % specify the gpu device

% ------------- dnCNN Baseline ----------------------
% Check project_generator for more configuration items
opts.use_cache = true;

opts.noise.type = 'gaussian';
opts.noise.std = 25;

opts.net.depth = 17;
opts.net.width = 64;
opts.net.relutype = 'relu';

opts.net.loss_function = 'mse';

opts.train.GradientThreshold = 0.005;
opts.train.GradientThresholdMethod = 'l2norm';
opts.train.MaxEpochs = 50; 

opts.train.InitialLearnRate = 0.1;
opts.train.LearnRateDropFactor = 0.87;


% ----------- Run Project(s) ------------
% overwrite configurations here

train_depth(opts,17);
train_depth(opts,16);
train_depth(opts,18);
train_depth(opts,19);
train_depth(opts,20);

train_relu(opts,'relu');
train_relu(opts,'leaky');
train_relu(opts,'clipped');



function train_depth(opts,depth)
  opts.net.depth = depth;

  opts.noise.std = 5;
  Project.run_project(opts);

  opts.noise.std = 25;
  Project.run_project(opts);

  opts.noise.std = 50;
  Project.run_project(opts);
end


function train_relu(opts,relu)
  opts.net.relutype = relu;
  Project.run_project(opts);
end

function train_lr(opts,lr,dropfactor)
  opts.train.LearnRateDropFactor = dropfactor;
  
end




