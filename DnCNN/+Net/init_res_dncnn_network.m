function lgraph = init_res_dncnn_network(image_size,varargin)
    % Initialize the layers of DnCNN network proposed by [1]
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
    %     Number of convolutional layers of the whole network
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
    %
    % Usage:
    %   Net.DnCNN_init_model(image_size)
    %   Net.DnCNN_init_model(image_size,Name,Value)
    %
    % Exception prj.msgID.init:
    %   'DnCNN:init'
    %
    % Note:
    %   1. This network use VGG architecture, proposed by [3]
    %   2. Weight initialization use the method proposed by [2]
    %
    % References:
    %   [1] Zhang, K., Zuo, W., Chen, Y., Meng, D., & Zhang, L. (2017). Beyond a Gaussian denoiser: Residual learning of deep CNN for image denoising. IEEE Transactions on Image Processing, 26(7), 3142–3155. https://doi.org/10.1109/TIP.2017.2662206
    %   [2] He, K., Zhang, X., Ren, S., & Sun, J. (2015). Delving Deep into Rectifiers : Surpassing Human-Level Performance on ImageNet Classification. ICCV. https://doi.org/10.1.1.725.4861
    %   [3] Simonyan, K., & Zisserman, A. (2014). Very Deep Convolutional Networks for Large-Scale Image Recognition, 1–14. https://doi.org/10.1016/j.infsof.2008.09.005
    
    
    % global settings -- no need to change unless necessary
    conv_size = [3,3]; % size of convolutional kernel
    crelu_ceil = 10; % threshold of clipped ReLU layer
    lrelu_theta = 0.01; % scale of leaky ReLU layer
    msgID = 'DnCNN:init';
    
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
        error(prj.msgID.init,err_msg);
    end
    
    % loss function
    switch string(options.loss_function)
      case "mse"
        get_regressionlayer = @() regressionLayer;
        options.loss_function = 'mse_reg'; % for create layer name
      otherwise
        err_msg = 'loss_function should be one of the following: "mse"';
        error(msgID,err_msg);
    end
    
    net_depth = options.net_depth;
    net_width = options.net_width;
    image_size = options.image_size;
    num_color_channels = image_size(3); % 1 for gray image, 3 for RGB image

    
    
    % define layers
    % input
%     input_layer = imageInputLayer(image_size,...
%       'Normalization','zerocenter');
    input_layer = imageInputLayer(image_size,...
      'Normalization','none');
    % res block
    res_layers = get_res_layers(conv_size,net_width,get_relulayer);
    % output
    output_layer = [...
      convolution2dLayer(conv_size,num_color_channels,...
                'NumChannels',net_width,...
                'Padding','same');...
      get_regressionlayer()];
    
    
    % build up layers
    num_res = (net_depth - 1)/2;
    layers = [...
      input_layer;...
      repmat(res_layers,num_res,1);...
      output_layer;];


    % add layer names in order to create layerGraph
    layers(1).Name = 'input';
    begin_idx = length(input_layer) + 1;
    end_idx = begin_idx + num_res*length(res_layers);
    for i = 1:num_res
      cur_idx = begin_idx + (i-1)*length(res_layers);
      layers(cur_idx).Name = strcat('conv',num2str(cur_idx));
      layers(cur_idx+1).Name = strcat('bn',num2str(cur_idx+1));
      layers(cur_idx+2).Name = strcat(options.relu_type,num2str(cur_idx+2));
      layers(cur_idx+3).Name = strcat('conv',num2str(cur_idx+3));
      layers(cur_idx+4).Name = strcat('add',num2str(cur_idx+4));
      layers(cur_idx+5).Name = strcat('bn',num2str(cur_idx+5));
      layers(cur_idx+6).Name = strcat(options.relu_type,num2str(cur_idx+6));
    end
    layers(end_idx).Name = strcat('conv',num2str(end_idx));
    layers(end).Name = options.loss_function;
    
    % create layerGraph in order to make skip connection
    lgraph = layerGraph(layers);
    
    % connect layers
    for i = 1:num_res
      cur_idx = begin_idx + (i-1)*length(res_layers);
      lgraph = connectLayers(lgraph,layers(cur_idx).Name,strcat(layers(cur_idx+4).Name,'/in2'));
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
      @(x)validateattributes(x,{'numeric','odd'},{'>=',2}));
    p.addParameter('net_width',defaults.net_width,...
      @(x)validateattributes(x,{'numeric'},{'>=',1}));
    p.addParameter('relu_type',defaults.relu_type,...
      @(x)any(validatestring(x,{'relu','leaky','clipped'})));
    p.addParameter('loss_function',defaults.loss_function,...
      @(x)any(validatestring(x,{'mse'})));
    
    
    % parse and deal inputs
    p.parse(varargin{:});
    options = p.Results;
    
    % input image size
    switch length(p.Results.image_size)
      case 3
        options.image_size = p.Results.image_size;
      case 2
        options.image_size = p.Results.image_size;
        options.image_size(3) = 1;
      case 1
        options.image_size = [p.Results.image_size,p.Results.image_size,1];
      otherwise
        err_msg = 'image_size should be row vector of three integer values';
        error(msgID,err_msg);
    end
end

function res_layers = get_res_layers(conv_size,net_width,varargin)
  p = inputParser;
  p.addOptional('get_relu_func',reluLayer);
  p.parse(varargin{:});
  get_relulayer = p.Results.get_relu_func;
  
  res_layers = [...
        convolution2dLayer(conv_size,net_width,...
            'NumChannels',net_width,...
            'Padding','same');...
        batchNormalizationLayer();...
        get_relulayer();
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
end
