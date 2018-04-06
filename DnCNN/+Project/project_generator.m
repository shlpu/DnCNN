function prj = project_generator(src_prj,varargin)
  
  defaults = get_defaults();
  prj = update_info(defaults,src_prj);
  
  p = inputParser();
  p.addParameter('traindata',prj.path.train);
  p.addParameter('testdata',prj.path.test);
  
  p.parse(varargin{:});
  

  % path setting
  prj.path.train =  p.Results.traindata;
  prj.path.test = p.Results.testdata;
  

  if prj.train.Plots
    prj.train.Plots = 'training-progress';
  else
    prj.train.Plots = 'none';
  end
  
  if prj.imds.use_augmentation
    prj.imds.augmenter = imageDataAugmenter( ...
        'RandXReflection',true, ...
        'RandXScale',[0.7 1], ...
        'RandYReflection',true, ...
        'RandYScale',[0.7 1],...
        'RandRotation',[0,360]);
  else
    prj.imds.augmenter = 'none';
  end
  
  
  
  % prevent a bug introduced by R2018a
  if ~mod(prj.imds.PatchesPerImage, prj.train.MiniBatchSize) == 0
    error('MATLAB:bug','PatchesPerImage should be divided by MiniBatchSize');
  end
  
  % update project information
  prj = update_path_info(prj);
  prj.noise = reduce_noise_info(prj.noise);
  
  % create project folder
  create_prj_folder(prj.path,prj.use_cache);
  
end

function defaults = get_defaults()
  % default project settings
  
  prj.use_cache = true; % keep all the existing trained nets of the current project
  
  % path setting
  prj.path.train = fullfile('Data','Train400');
  prj.path.test.BSD68 = fullfile('Data','Test','BSD68');
  prj.path.test.Set12 = fullfile('Data','Test','Set12');
  prj.path.test.Set5 = fullfile('Data','Test','Set5');
  
  % gaussian noise setting
  prj.noise.type = 'gaussian'; % gaussian | impulse
  prj.noise.std = 25; % for gaussian noise
  prj.noise.Density = 0.05;
  prj.noise.ValueRange = [0,1];
  
  % imagedatastore setting
  
  prj.imds.PatchSize = 40;
  prj.imds.PatchesPerImage = 512;
  prj.imds.use_augmentation = true;
    
    
  % net
  prj.net.type = 'vgg'; % vgg | res
  prj.net.depth = 17;
  prj.net.width = 64;
  prj.net.relutype = 'leaky'; % relu | clipped | leaky
  prj.net.loss_function = 'mse'; % mse 
  prj.net.ssim_sigma = 5;

  
  % training setting
  prj.train.solver = 'adam'; % sgdm | adam
  
  prj.train.GradientThreshold = inf;
  prj.train.GradientThresholdMethod = 'l2norm'; % l2norm | global-l2norm
  prj.train.MaxEpochs = 50; 
  
  prj.train.MiniBatchSize = 128;
  
  prj.train.InitialLearnRate = 1e-3;
  prj.train.LearnRateDropFactor = 0.1;
  prj.train.LearnRateDropPeriod = 30;
  
  prj.train.Verbose = true; 
  prj.train.Plots = false;
  prj.train.VerboseFrequency = 5;
  
  defaults = prj;
end


function dest_info = update_info(defaults,src_info)
  % update custom info according to defaults info
  dest_info = defaults;
  if isstruct(src_info)
    config_lists = fieldnames(defaults);
    names = fieldnames(src_info);
    for i = 1:length(names)
      if ~any(strcmp(names{i},config_lists)) % check if this is new item
        warning('new configuration item %s added',names{i});
      end
      eval_code = sprintf(...
          'dest_info.%s = update_info(defaults.%s, src_info.%s); '...
          ,names{i},names{i},names{i});
        eval(eval_code);
    end
  else
    dest_info = src_info;
  end
end


function create_prj_folder(pathinfo,use_cache)
% create project folder
  if ~use_cache 
    Utilities.rebuild_dir(pathinfo.root);
  end
  
  if ~exist(pathinfo.root, 'dir')
      mkdir(pathinfo.root);
      mkdir(pathinfo.Checkpoint);
      structfun(@mkdir,pathinfo.results);
  end
end

function dest_info = update_path_info(src_info)
  dest_info = src_info;
  
  noise_foldername = get_noise_foldername(dest_info.noise);
  
  dest_info.path.root = fullfile('Results',...
    strcat(dest_info.net.type,'_',dest_info.net.loss_function),...
    noise_foldername,...
    strcat('depth',num2str(dest_info.net.depth),'_width',num2str(dest_info.net.width),'_',dest_info.net.relutype),...
    strcat(dest_info.train.solver),...
    strcat('learnrate',num2str(dest_info.train.InitialLearnRate),'_drop',num2str(dest_info.train.LearnRateDropFactor),'_period',num2str(dest_info.train.LearnRateDropPeriod)),...
    strcat('gradclip_',dest_info.train.GradientThresholdMethod,'_',num2str(dest_info.train.GradientThreshold)));
  
  dest_info.path.Checkpoint = fullfile(dest_info.path.root,'Checkpoints');
  
  dest_info.path.results = structfun(@(x)get_result_path(x,dest_info.path.root), dest_info.path.test,'UniformOutput',false);
  
  dest_info.prefix.net = fullfile(dest_info.path.Checkpoint,'epoch_');

  dest_info.path.log.performance = fullfile(dest_info.path.root,'performance.txt');
  
  dest_info.path.log.training = fullfile(dest_info.path.root,'training.txt');
end


function result_path = get_result_path(datapath,rootpath)
  [~,dataname] = fileparts(datapath);
  result_path = fullfile(rootpath,'imgs',dataname);
end


function noise_foldername = get_noise_foldername(noiseinfo)
  switch string(noiseinfo.type)
    case "gaussian"
      noise_foldername = strcat(noiseinfo.type,'_std',num2str(noiseinfo.std));
    case "impulse"
      if numel(noiseinfo.ValueRange) == 2
        noise_foldername = strcat('salt&pepper_density',num2str(noiseinfo.Density));
      else
        noise_foldername = strcat('random_impulse_density',num2str(noiseinfo.Density));
      end
    otherwise
      error('Training:train_network','Unsupported noise type, should be : gaussian, impulse');
  end
end

function dest_info = reduce_noise_info(src_info)
  dest_info.type = src_info.type;
  switch string(src_info.type)
    case "gaussian"
      dest_info.std = src_info.std;
    case "impulse"
      dest_info.Density = src_info.Density;
      dest_info.ValueRange = src_info.ValueRange;
    otherwise
      error('Training:train_network','Unsupported noise type, should be : gaussian, impulse');
  end
end
