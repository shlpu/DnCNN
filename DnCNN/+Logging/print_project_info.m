function print_project_info(prj, varargin)
% print project information into logfile
%
% Parameters:
%   prj (Required)
%   logfile (Parameter)


  Logging.print_line('Project Info','filler','=',varargin{:});
  
  Logging.print_line('Noise Info',varargin{:});
  Logging.print_info(prj.noise,varargin{:});
  
  Logging.print_line('Training Options',varargin{:});
  Logging.print_info(prj.train,varargin{:});
  
  Logging.print_line('Net Info',varargin{:});
  Logging.print_info(prj.net,varargin{:});
  
  Logging.print_line('Training Data Info',varargin{:});
  Logging.print_info(prj.imds,varargin{:});
  
  Logging.print_line('Path Info',varargin{:});
  Logging.print_info(prj.path,varargin{:});
  
  Logging.print_line('GPU Info',varargin{:});
  Logging.print_info(gpuDevice,varargin{:});
  
  Logging.print_line('',varargin{:},'filler','=');


  
end


  