function log_errors(opts,varargin)
% called when an error is thrown during the project
% log error and project information for debug
% 
% Parameters:
%   opts  (Required) : structure
%     project configurations and information
%   logfile (Parameter) : string | char array
%
% Example 1:
%   try
%     net = Training.train_network(opts);
%   catch
%     Logging.log_errors(opts,'logfile','error.txt');
%     net = [];
%     return;
%   end
% 
    
    % remove irrelevant information  
    opts = rmfield(opts,'path');
    opts = rmfield(opts,'prefix');
    opts.train = rmfield(opts.train,'Verbose');
    opts.train = rmfield(opts.train,'VerboseFrequency');
    opts.train = rmfield(opts.train,'Plots');
    
    
    err_msg = string(lasterr); %#ok<LERR>
    
    
    Logging.print_line('Error Info',varargin{:});
    Logging.print(err_msg,varargin{:});
    Logging.print('\n',varargin{:});
    
    Logging.print_line('Noise Info',varargin{:});
    Logging.print_info(opts.noise,varargin{:});

    Logging.print_line('Training Options',varargin{:});
    Logging.print_info(opts.train,varargin{:});

    Logging.print_line('Net Info',varargin{:});
    Logging.print_info(opts.net,varargin{:});
    
    Logging.print_line('',varargin{:},'filler','=');
end