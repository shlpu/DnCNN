function net = load_net(path,varargin)
  % load specified network or the latest network
  % if given path is a .mat file, then the file will be loaded,
  % if given path is a folder, then the latest .mat file will be loaded.
  %
  % Return:
  %   net : Network
  % 
  % Parameters:
  %   path (Required): char array | string
  %     path to the network or folders of network
  %   netname (Optional): char array | string (default 'net')
  %     variable name of the network
  
  p = inputParser();
  p.addOptional('netname','net',@(x) ischar(x) || isstring(x));
  p.parse(varargin{:});
  
  netname = p.Results.netname;
  
  if isfolder(path)
    net = load_latest_net(path,netname);
  else
    net = load(path,netname); %#ok<NASGU>
    net = eval(['net.',netname]);
  end
end

function net = load_latest_net(path,netname)
  net_list = dir(fullfile(path,'*.mat'));
  if ~isempty(net_list)
    % load the latest checkpoint network
    net_files = {net_list.name};
    net_dates = [net_list.datenum];
    
    % load latest net
    [~,max_index] = max(net_dates);
    latest_net_file = net_files{max_index};
    net = Utilities.load_net(fullfile(path,latest_net_file),netname);
  else
    error('No network found');
  end
end