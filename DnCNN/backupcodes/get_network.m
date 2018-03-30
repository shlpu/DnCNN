function [net, iteration] = get_network(checkpoint_path,init_func,varargin)
% Load the latest cache network if there is,
% if there isn't, then initialize a new network using init_func
%
%
% Parameters:
%   checkpoint_path (Required) : char array | string
%     specify the location where net work caches are stored
%   init_func (Required) : function handle
%     initialize network using init_func(A,B,...) when no 
%     net cache is avaliable
%   A,B,... (Optional)
%     parameters that will be passed to init_func
%     
% Return:
%   net : Serial Network
% 
% Usage:
%   net = get_network(checkpoint_path, init_func);
%   net = get_network(checkpoint_path, init_func, A, B,...);
% 
% Note:
%   1. the name of the cache network should satify the following format
%       "convnet_checkpoint__ITERATION__YYYY_MM_DD__HH_MM_SS.mat"
%     e.g.,
%       "convnet_checkpoint__100__2018_03_26__19_30_09.mat"
%   2. the latest checkpoint is decided by its time
%   3. the iteration is only a reference to calculate the epoch
  
  if ~exist(checkpoint_path,'dir')
    mkdir(checkpoint_path);
  end
  
  net_list = dir(fullfile(checkpoint_path,'*.mat'));
  if ~isempty(net_list)
    % load the latest checkpoint network
    net_files = {net_list.name};
    [net_iters,net_dates] = cellfun(@(s) parse_name(s),net_files);
    
    % calculate overall iterations
    [~,index] = sort(net_dates);
    net_iters = net_iters(index);
    iteration = sum(net_iters(diff(net_iters)<0)) + net_iters(end);
    
    % load latest net
    [~,max_index] = max(net_dates);
    latest_net_file = net_files{max_index};
    net = Utilities.load_net(fullfile(checkpoint_path,latest_net_file),'net');
  else
    % initialize a new network
    net = init_func(varargin{:});
    iteration = 0;
  end
  
end

function [net_iter,net_date] = parse_name(netname)
% parse the name of the checkpoint file
%
% Return:
%   iteration : integer
%   date : datetime
%
% Parameters:
%   netname (Required) : char array | string
%     name of the checkpoint net
%
% Note:
%   1. netname should satify the following format
%       "convnet_checkpoint__ITERATION__YYYY_MM_DD__HH_MM_SS.mat"
%     e.g.,
%       "convnet_checkpoint__100__2018_03_26__19_30_09.mat"

  ITER_PREFIX = 'checkpoint__';
  ITER_REG = strcat(ITER_PREFIX,'\d*');
  DATE_REG = '\d{4}_\d{2}_\d{2}__\d{2}_\d{2}_\d{2}';
  
  % parse iteration
  [startIndex,endIndex] = regexp(netname,ITER_REG);
  net_iter = str2num(netname(startIndex+numel(ITER_PREFIX):endIndex)); %#ok<ST2NM>
  
  % parse date
  [startIndex,endIndex] = regexp(netname,DATE_REG);
  date_str = netname(startIndex:endIndex);
  net_date = datetime(date_str,'InputFormat','yyyy_MM_dd__HH_mm_ss');
end