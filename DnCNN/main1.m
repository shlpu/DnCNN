clear all;
gpuDevice(1); % specify the gpu device


% ------------- Baseline Setting ----------------------
% Check project_generator for more configuration items
opts.use_cache = true;

opts.noise.type = 'gaussian'; % gaussian | impulse
opts.noise.std = 25;
opts.noise.Density = 0.05;

opts.net.type = 'res'; % vgg | res
opts.net.depth = 17;
opts.net.width = 64;
opts.net.relutype = 'relu';

opts.net.loss_function = 'ssim'; % ssim | mse

opts.train.solver = 'adam';

opts.train.GradientThreshold = inf;
opts.train.GradientThresholdMethod = 'l2norm';
opts.train.MaxEpochs = 80;

opts.train.InitialLearnRate = 1e-3;
opts.train.LearnRateDropFactor = 0.1;
opts.train.LearnRateDropPeriod = 30;




% ----------- Run Project(s) ------------


%  train_lr(opts,0.001, 1, 10)
%  train_lr(opts,0.01, 0.1, 10)
%  train_lr(opts,1e-3, 0.1,30)
%  train_lr(opts,0.1, 0.1, 10)



opts.train.InitialLearnRate = 1e-3;
opts.train.LearnRateDropFactor = 0.1;
opts.train.LearnRateDropPeriod = 30;
train_depth_width(opts,17,64);



% ----------------- Helpers ---------------------


function train_depth_width(opts,depth,width)
  opts.net.depth = depth;
  opts.net.width = width;
  Project.run_project(opts);
end

function train_lr(opts,lr,dropfactor,dropperiod)
  opts.train.LearnRateDropFactor = dropfactor;
  opts.train.InitialLearnRate = lr;
  opts.train.LearnRateDropPeriod = dropperiod;
  
  Project.run_project(opts);
end


