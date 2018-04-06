function lgraph = init_dncnn_network(image_size,varargin)
    % Initialize the layers of DnCNN network proposed by [1]
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
    %       'ssim'             -- structure similarity index
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
      case "ssim"
        get_regressionlayer = @() Net.Layer.SSIMRegressionLayer(options.ssim_sigma);
        options.loss_function = 'ssim_reg';
      otherwise
        err_msg = 'loss_function should be one of the following: "mse, SSIM"';
        error(msgID,err_msg);
    end
    
    net_depth = options.net_depth;
    net_width = options.net_width;
    image_size = options.image_size;
    num_color_channels = image_size(3); % 1 for gray image, 3 for RGB image

    
    
    % define layers
    % input layer
%     input_layer = imageInputLayer(image_size,...
%       'Normalization','zerocenter');
    input_layer = imageInputLayer(image_size,...
      'Normalization','none');

    % Layer type 1: conv+ReLu
    L1 = [...
        convolution2dLayer(conv_size,net_width,...
            'NumChannels',num_color_channels,...
            'Padding','same');...
        get_relulayer()];
    L1(1).Weights = randn([conv_size,num_color_channels,net_width]) .* sqrt(2/(prod(conv_size)*net_width));
    L1(1).Bias = zeros([1,1,net_width]);
    
    % Layer type 2: conv+BN+ReLu
    L2 = [...
        convolution2dLayer(conv_size,net_width,...
            'NumChannels',net_width,...
            'Padding','same');...
        batchNormalizationLayer();...
        get_relulayer()];
    L2(1).Weights = randn([conv_size,net_width,net_width]) .*sqrt(2/(prod(conv_size)*net_width));
    L2(1).Bias = zeros([1,1,net_width]);
    
    % Layer type 3: conv
    L3 = convolution2dLayer(conv_size,num_color_channels,...
            'NumChannels',net_width,...
            'Padding','same');
    L3(1).Weights = randn([conv_size,net_width,num_color_channels]) .*sqrt(2/(prod(conv_size)*net_width));
    L3(1).Bias = zeros([1,1,num_color_channels]);
    
    
    
    % initializing network
    num_L2 = net_depth-2;
    layers = [...
        input_layer;...
        L1;...
        repmat(L2, num_L2,1);...
        L3;...
        get_regressionlayer()];
      
    % add layer names
    layers(1).Name = 'input';
    layers(2).Name = 'conv1';
    layers(3).Name = strcat(options.relu_type,'2');
    begin_idx = length(input_layer) + length(L1);
    end_idx = begin_idx + num_L2*length(L2);
    for i = 1:num_L2
      cur_idx = begin_idx + (i-1)*length(L2) + 1;
      layers(cur_idx).Name = strcat('conv',num2str(cur_idx));
      layers(cur_idx+1).Name = strcat('bn',num2str(cur_idx+1));
      layers(cur_idx+2).Name = strcat(options.relu_type,num2str(cur_idx+2));
    end
    layers(end_idx+1).Name = strcat('conv',num2str(end_idx+1));
    
    layers(end).Name = strcat(options.loss_function);
    
    lgraph = layerGraph(layers);
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
