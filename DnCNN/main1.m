clear all;
gpuDevice(1); % specify the gpu device


% ------------- Baseline Setting ----------------------
% Check project_generator for more configuration items
opts.use_cache = true;

opts.noise.type = 'gaussian';
opts.noise.std = 25;

opts.net.depth = 17;
opts.net.width = 64;
opts.net.relutype = 'leaky';

opts.net.loss_function = 'mse';

opts.train.GradientThreshold = 0.005;
opts.train.GradientThresholdMethod = 'l2norm';
opts.train.MaxEpochs = 1; 

opts.train.InitialLearnRate = 0.1;
opts.train.LearnRateDropFactor = 0.1;
opts.train.LearnRateDropPeriod = 10;


% ----------- Run Project(s) ------------

% train_lr(opts,0.1, 0.87, 1) % PSNR 28.01
train_lr(opts,0.01, 0.87, 1) % PSNR 26.45
% train_lr(opts,0.001, 0.87, 1) % PSNR 19.30
% train_lr(opts,0.0001, 0.87, 1) % 


 train_lr(opts,0.001, 1, 10)
 train_lr(opts,0.01, 0.1, 10)
%  train_lr(opts,0.0001, 1, 10)
%  train_lr(opts,0.1, 0.1, 10)



% train_depth_width(opts,17,64);
% train_depth_width(opts,16,64);
% train_depth_width(opts,18,64);
% train_depth_width(opts,19,64);
% train_depth_width(opts,20,64);

% train_depth_width(opts,17,128);
% train_depth_width(opts,16,128);
% train_depth_width(opts,18,128);
% train_depth_width(opts,19,128);
% train_depth_width(opts,20,128);
% 
% train_relu(opts,'relu');
% train_relu(opts,'leaky');
% train_relu(opts,'clipped');



% ----------------- Helpers ---------------------


function train_depth_width(opts,depth,width)
  opts.net.depth = depth;
  opts.net.width = width;
  Project.run_project(opts);
end


function train_relu(opts,relu)
  opts.net.relutype = relu;
  Project.run_project(opts);
end

function train_lr(opts,lr,dropfactor,dropperiod)
  opts.train.LearnRateDropFactor = dropfactor;
  opts.train.InitialLearnRate = lr;
  opts.train.LearnRateDropPeriod = dropperiod;
  
  Project.run_project(opts);
end




