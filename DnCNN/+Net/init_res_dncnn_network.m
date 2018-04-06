function lgraph = init_res_dncnn_network(image_size,varargin)
    % Initialize the layers of DnCNN network proposed by [1] and [3]
    % with architecture replaced by resnet
    %
    % Return:
    %   lgraph:  nnet.cnn.LayerGraph
    %   
    % Parameters:
    %   image_size: (Required) row vector of three integer values
    %     Size of input images of the whole network
    %     Example: [128 128 3]
    %
    %   net_depth: (Parameter) integer (default 17)
    %     Number of convolutional layers of the whole network, should be 3n+2
    %
    %   net_width: (Parameter) integer (default 64)
    %     Number of filters of each convolutional layer
    %
    %   relu_type: (Parameter) char vector | string
    %     Using specific ReLU layers in the network
    %       'relu'(default)    -- normal ReLU layer
    %       'leaky'            -- leaky ReLU layer with scale of 0.01
    %       'clipped'          -- clipped ReLU layer with threshold of 10
    %   loss_function: (Parameter) char vector | string
    %     Using specific loss function for regression layer
    %       'mse'(default)     -- mean-squared-error
    %       'ssim'             -- structure similarity index
    %
    % Usage:
    %   Net.DnCNN_init_model(image_size)
    %   Net.DnCNN_init_model(image_size,Name,Value)
    %
    %
    % Note:
    %   1. This network use resnet architecture, proposed by [3] and [4]
    %   2. Weight initialization use the method proposed by [2]
    %
    % References:
    %   [1] Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Transactions on Image Processing, 26(7), 3142–3155. https://doi.org/10.1109/TIP.2017.2662206
    %   [2] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers : Surpassing Human-Level Performance on ImageNet Classification. ICCV. https://doi.org/10.1.1.725.4861
    %   [3] Bae, W., Yoo, J., & Ye, J. C. (2017). Beyond Deep Residual Learning for Image Restoration: Persistent Homology-Guided Manifold Simplification. IEEE Computer Society Conference on Computer Vision and Pattern Recognition Workshops, 2017–July, 1141–1149. https://doi.org/10.1109/CVPRW.2017.152    
    %   [4] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. Multimedia Tools and Applications, 1–17. Retrieved from http://arxiv.org/abs/1512.03385

    % global settings -- no need to change unless necessary
    conv_size = [3,3]; % size of convolutional kernel
    crelu_ceil = 10; % threshold of clipped ReLU layer
    lrelu_theta = 0.01; % scale of leaky ReLU layer
    
    options = parse_input(image_size,varargin{:});
    
    % ReLU type
    switch string(options.relu_type)
      case "relu"
        get_relulayer = @(varargin) reluLayer(varargin{:});
      case "clipped"
        get_relulayer = @(varargin) clippedReluLayer(crelu_ceil,varargin{:});
        options.relu_type = "clipped_relu"; % for create layer name
      case "leaky"
        get_relulayer = @(varargin) leakyReluLayer(lrelu_theta,varargin{:});
        options.relu_type = "leaky_relu"; % for create layer name
      otherwise
        err_msg = 'relu_type should be one of the following: "relu","leaky","clipped"';
        error('Net:init_res_dncnn_network',err_msg);
    end
    
    % loss function
    switch string(options.loss_function)
      case "mse"
        get_regressionlayer = @() regressionLayer;
        options.loss_function = 'mse_reg'; % for create layer name
      case "ssim"
        get_regressionlayer = @() Net.Layer.SSIMRegressionLayer(options.ssim_sigma);
        options.loss_function = 'ssim_reg';
      otherwise
        err_msg = 'loss_function should be one of the following: "mse, SSIM"';
        error('Net:init_res_dncnn_network',err_msg);
    end
    
    net_depth = options.net_depth;
    net_width = options.net_width;
    image_size = options.image_size;
    num_color_channels = image_size(3); % 1 for gray image, 3 for RGB image

    
    % build up layers
    num_res = (net_depth - 2)/3;
    
    input_layers = get_input_layers(image_size,conv_size,net_width,get_relulayer);
    res_layers = get_res_layers(conv_size,net_width,get_relulayer);
    layers = [...
      input_layers;
      
      repmat(res_layers,num_res,1);

      convolution2dLayer(conv_size,num_color_channels,...
                    'NumChannels',net_width,...
                    'Padding','same');...
      get_regressionlayer();
      ];


    % add layer names in order to create layerGraph
    layers(1).Name = 'input';
    layers(2).Name = 'conv1';
    layers(3).Name = 'relu2';
    begin_idx = length(input_layers) + 1;
    end_idx = begin_idx + num_res*length(res_layers);
    for i = 1:num_res
      cur_idx = begin_idx + (i-1)*length(res_layers);
      layers(cur_idx).Name = strcat('conv',num2str(cur_idx-1));
      layers(cur_idx+1).Name = strcat('bn',num2str(cur_idx));
      layers(cur_idx+2).Name = strcat('relu',num2str(cur_idx+1));
      layers(cur_idx+3).Name = strcat('conv',num2str(cur_idx+2));
      layers(cur_idx+4).Name = strcat('bn',num2str(cur_idx+3));
      layers(cur_idx+5).Name = strcat('relu',num2str(cur_idx+4));
      layers(cur_idx+6).Name = strcat('conv',num2str(cur_idx+5));
      layers(cur_idx+7).Name = strcat('add',num2str(cur_idx+6));
      layers(cur_idx+8).Name = strcat('bn',num2str(cur_idx+7));
      layers(cur_idx+9).Name = strcat('relu',num2str(cur_idx+8));
    end
    layers(end_idx).Name = strcat('conv',num2str(end_idx-1));
    layers(end).Name = 'mse_reg';
    
    % create layerGraph in order to make skip connection
    lgraph = layerGraph(layers);
    for i = 1:num_res
      cur_idx = begin_idx + (i-1)*length(res_layers);
      lgraph = connectLayers(lgraph,layers(cur_idx-1).Name,strcat(layers(cur_idx+7).Name,'/in2'));
    end
    
    
end

function options = parse_input(varargin)
% default parameter value
    defaults.net_depth = 17;
    defaults.net_width = 64;
    defaults.relu_type = 'relu';
    defaults.loss_function = 'mse';

    % add inputs constrain
    p = inputParser();
    p.addRequired('image_size',@isnumeric);
    p.addParameter('net_depth',defaults.net_depth,...
      @(x)validateattributes(x,{'numeric'},{'>=',2}));
    p.addParameter('net_width',defaults.net_width,...
      @(x)validateattributes(x,{'numeric'},{'>=',1}));
    p.addParameter('relu_type',defaults.relu_type,...
      @(x)any(validatestring(x,{'relu','leaky','clipped'})));
    p.addParameter('loss_function',defaults.loss_function,...
      @(x)any(validatestring(x,{'mse','ssim'})));
    p.addParameter('ssim_sigma',5,@isnumeric);
    
    
    % parse and deal inputs
    p.parse(varargin{:});
    options = p.Results;
    
    % input image size
    switch length(options.image_size)
      case 3
        options.image_size = p.Results.image_size;
      case 2
        options.image_size = p.Results.image_size;
        options.image_size(3) = 1;
      case 1
        options.image_size = [p.Results.image_size,p.Results.image_size,1];
      otherwise
        err_msg = 'image_size should be row vector of three integer values';
        error('Net:init_res_dncnn_network',err_msg);
    end
    
%     % check net depth
    if mod(options.net_depth-2,3)
      err_msg = 'resnet architecture support net_depth of only 3n+2';
      error('Net:init_res_dncnn_network',err_msg);
    end
end

function input_layers = get_input_layers(image_size,conv_size,net_width,varargin)
  p = inputParser;
  p.addOptional('get_relu_func',reluLayer);
  p.parse(varargin{:});
  get_relulayer = p.Results.get_relu_func;
  
  input_layers = [...
    imageInputLayer(image_size,...
        'Normalization','zerocenter');
    convolution2dLayer(conv_size,net_width,...
        'NumChannels',net_width,...
        'Padding','same');
    get_relulayer();
    ];

%   % Weight initialization would raise an cuDNN error in matlab R2018a
%   input_layers(2).Weights = randn([conv_size,net_width,net_width]) .*sqrt(2/(prod(conv_size)*net_width));
%   input_layers(2).Bias = zeros([1,1,net_width]);
end

function res_layers = get_res_layers(conv_size,net_width,varargin)
  p = inputParser;
  p.addOptional('get_relu_func',reluLayer);
  p.parse(varargin{:});
  get_relulayer = p.Results.get_relu_func;
  
  % define resnet module
  res_layers = [...
    convolution2dLayer(conv_size,net_width,...
        'NumChannels',net_width,...
        'Padding','same');...
    batchNormalizationLayer();...
    get_relulayer();...

    convolution2dLayer(conv_size,net_width,...
        'NumChannels',net_width,...
        'Padding','same');...
    batchNormalizationLayer();...
    get_relulayer();...

    convolution2dLayer(conv_size,net_width,...
      'NumChannels',net_width,...
      'Padding','same');...

    additionLayer(2);...

    batchNormalizationLayer();...
    get_relulayer();];
  
  res_layers(1).Weights = randn([conv_size,net_width,net_width]) .*sqrt(2/(prod(conv_size)*net_width));
  res_layers(1).Bias = zeros([1,1,net_width]);
  res_layers(4).Weights = randn([conv_size,net_width,net_width]) .*sqrt(2/(prod(conv_size)*net_width));
  res_layers(4).Bias = zeros([1,1,net_width]);
  res_layers(7).Weights = randn([conv_size,net_width,net_width]) .*sqrt(2/(prod(conv_size)*net_width));
  res_layers(7).Bias = zeros([1,1,net_width]);
end
