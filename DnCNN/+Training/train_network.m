function net = train_network(prj)
  % train DnCNN network
  %
  % Return:
  %   net : Serial Network
  %
  % Parameters:
  %   prj : structure tree
  %     project options, containing the following parts of information
  %       1. path -------------------- path information 
  %         * train                 -- path to training data  
  %         * test  (structure)     -- contain paths to test data
  %         * Checkpoint            -- path to training checkpoints
  %         * results (structure)   -- contain paths to results of test data
  %         * log    (structure)    -- contain paths to log files
  %       2. imds -------------------- imageDatastore information
  %         * PatchSize             -- Image size for network input
  %         * PatchesPerImage       -- number of patches generated from each image
  %       3. training ---------------- trainingOption information
  %       4. log --------------------- ( Not Implemented yet)
  %       5. noise -------------------
  %       6. use_cache --------------- false to train a totally new network
  %           
  
  % global setting
  msgID = 'DnCNN:train';
  
  % load training datastore
  imds = load_training_datastore(prj.path.train,prj.imds,prj.noise,...
    'DataAugmentation',prj.imds.augmenter);
  
  % init network
  switch string(prj.net.type)
    case "vgg"
      init_network = @Net.init_dncnn_network;
    case "res"
      init_network = @Net.init_res_dncnn_network;
    otherwise
      error('Training:train_network','networktype invalid');
  end
  lgraph = init_network(prj.imds.PatchSize,...
    'net_depth',prj.net.depth,...
    'net_width',prj.net.width,...
    'relu_type',prj.net.relutype,...
    'loss_function',prj.net.loss_function,...
    'ssim_sigma',prj.net.ssim_sigma);
  cur_lr = prj.train.InitialLearnRate;
  
  % train network
  % use for-loop training to save fully-trained network for evaluation for each epoch
  %   hence the MaxEpochs in trainingOptions is set to 1
  % This might be time-consuming since gather() works very slowly.
  for i = 1:prj.train.MaxEpochs
    % read trained cache if exists
    cur_cache_net = strcat(prj.prefix.net,num2str(i),'.mat');
    if exist(cur_cache_net,'file')
      net = Utilities.load_net(cur_cache_net,'net');
    else
      if i >= 2
        last_cache_net = strcat(prj.prefix.net,num2str(i-1),'.mat');
        if ~exist(last_cache_net,'file')
          error(msgID,[last_cache_net,' missing']);
        end
        net = Utilities.load_net(last_cache_net,'net');
        lgraph = layerGraph(net);
      end
      
      % set training option
      training_opts = trainingOptions(prj.train.solver,...
        'InitialLearnRate',cur_lr,...
        'GradientThreshold',prj.train.GradientThreshold,...
        'GradientThresholdMethod',prj.train.GradientThresholdMethod,...
        'MaxEpochs',1,...
        'MiniBatchSize',prj.train.MiniBatchSize,...
        'Shuffle','never',...
        'Plots',prj.train.Plots,...
        'ExecutionEnvironment','gpu',...
        'Verbose',prj.train.Verbose,...
        'VerboseFrequency',prj.train.VerboseFrequency);
      
      net = trainNetwork(imds, lgraph, training_opts);
      save(strcat(prj.prefix.net,num2str(i)),'net');
    end
    
    save('latest_net','net'); % for debug
    if ~mod(i,prj.train.LearnRateDropPeriod)
      cur_lr = cur_lr * prj.train.LearnRateDropFactor;
    end
  end
end


function imds = load_training_datastore(datapath,imdsinfo,noiseinfo,varargin)
  
  p = inputParser();
  p.addParameter('DataAugmentation','none');
  p.parse(varargin{:});
  augumenter = p.Results.DataAugmentation;
  
  imds = imageDatastore(datapath,...
    'IncludeSubfolders',true,...
    'FileExtensions',{'.jpg','.png','.bmp','.jpeg'});
  tmp = imds.preview();
  switch size(tmp,3)
    case 1
      imdsinfo.ChannelFormat = 'Grayscale';
    case 3
      imdsinfo.ChannelFormat = 'RGB';
    otherwise
      error(msgID,'Wrong input data');
  end
  
  switch string(noiseinfo.type)
    case "gaussian"
      imds = Dataloader.GaussianNoiseImageDatastore(imds,...
        'PatchesPerImage',imdsinfo.PatchesPerImage,...
        'DataAugmentation',augumenter,...
        'PatchSize',imdsinfo.PatchSize,...
        'GaussianNoiseLevel',(noiseinfo.std ./255),...
        'ChannelFormat',imdsinfo.ChannelFormat,...
        'DispatchInBackground',false);
    case "impulse"
      imds = Dataloader.ImpulseNoiseImageDatastore(imds,...
        'PatchesPerImage',imdsinfo.PatchesPerImage,...
        'DataAugmentation',augumenter,...
        'PatchSize',imdsinfo.PatchSize,...
        'NoiseDensity',noiseinfo.Density,...
        'NoiseValueRange',noiseinfo.ValueRange,...
        'ChannelFormat',imdsinfo.ChannelFormat,...
        'DispatchInBackground',false);
    otherwise
      error('Training:train_network','Unsupported noise type, should be : gaussian, impulse');
  end
end
