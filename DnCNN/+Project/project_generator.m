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
        'RandXScale',[0.5 2], ...
        'RandYReflection',true, ...
        'RandYScale',[0.5 2],...
        'RandRotation',[0,360]);
  else
    prj.imds.augmenter = 'none';
  end
  
  % prevent a bug introduced by R2018a
  if ~mod(prj.imds.PatchesPerImage, prj.train.MiniBatchSize) == 0
    error('MATLAB:bug','PatchesPerImage should be divided by MiniBatchSize');
  end
  
  
  % generate project information
  prj.path.root = fullfile('Results',...
    strcat(prj.noise.type,'_std',num2str(prj.noise.std),'_mean',num2str(prj.noise.mean)),...
    strcat('depth',num2str(prj.net.depth),'_width',num2str(prj.net.width),'_',prj.net.relutype),...
    strcat('learnrate',num2str(prj.train.InitialLearnRate),'_drop',num2str(prj.train.LearnRateDropFactor),'_period',num2str(prj.train.LearnRateDropPeriod)),...
    strcat('gradclip_',prj.train.GradientThresholdMethod,'_',num2str(prj.train.GradientThreshold)));
  prj.path.Checkpoint = fullfile(prj.path.root,'Checkpoints');
  
  prj.path.results = structfun(@(x)get_result_path(x,prj.path.root), prj.path.test,'UniformOutput',false);
  prj.prefix.net = fullfile(prj.path.Checkpoint,'epoch_');

  prj.path.log.performance = fullfile(prj.path.root,'performance.txt');
  prj.path.log.training = fullfile(prj.path.root,'training.txt');
  
  
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
  
  % noise setting
  prj.noise.type = 'gaussian'; % gaussian
  prj.noise.std = 25; % for gaussian noise
  prj.noise.mean = 0; % for gaussian noise
  
  % imagedatastore setting
  
  prj.imds.PatchSize = 50;
  prj.imds.PatchesPerImage = 512;
  prj.imds.use_augmentation = true;
    
    
  % net
  prj.net.depth = 17;
  prj.net.width = 64;
  prj.net.relutype = 'leaky'; % relu | clipped | leaky
  prj.net.loss_function = 'mse'; % mse 

  
  % training setting
  prj.train.solver = 'sgdm'; % sgdm
  prj.train.Momentum = 0.9; 
  
  prj.train.GradientThreshold = 0.005;
  prj.train.GradientThresholdMethod = 'l2norm'; % l2norm | global-l2norm
  prj.train.MaxEpochs = 50; 
  
  prj.train.MiniBatchSize = 128;
  
  prj.train.InitialLearnRate = 0.1;
  prj.train.LearnRateDropFactor = 0.87;
  prj.train.LearnRateDropPeriod = 1;
  
  prj.train.Verbose = true; 
  prj.train.Plots = false;
  prj.train.VerboseFrequency = 10;
  
  defaults = prj;
end


function dest_info = update_info(defaults,src_info)
  % update custom info according to defaults info
  dest_info = defaults;
  if isstruct(src_info)
    config_lists = fieldnames(defaults);
    names = fieldnames(src_info);
    for i = 1:length(names)
      if any(strcmp(names{i},config_lists)) % check if this is new item
        eval_code = sprintf(...
          'dest_info.%s = update_info(defaults.%s, src_info.%s); '...
          ,names{i},names{i},names{i});
        eval(eval_code);
      else
        warning('new configuration item %s added',names{i});
      end
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

function result_path = get_result_path(datapath,rootpath)
  [~,dataname] = fileparts(datapath);
  result_path = fullfile(rootpath,'imgs',dataname);
end
