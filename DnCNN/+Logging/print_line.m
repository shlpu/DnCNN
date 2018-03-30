function print_line(title,varargin)
% print(append) a line to log file or console
%
% Parameters:
%   title:  (Optional) char array | string (default '')
%     the title of the line, printed in the center
%   logfile: (Parameter) char array | string
%     log file path
%     example: "log.txt"
%   width:  (Parameter) integer larger and equal than 20 (default 80)
%     Specify how many characters a line should have.
%   filler: (Parameter) char (default '-')
%     the character used for filling the line
%
%     
% Usage:
%   print to console:
%     print_line(title,Name,Value); % print to console
%     print_line(title,'logfile',logfile,Name,Value); % append to the end of logfile


  % add parameter constrains
  defaults.logfile = "CONSOLE";
  defaults.width = 80;
  defaults.filler = '-';

  p = inputParser();
  p.addParameter('logfile',defaults.logfile, @(x)ischar(x)||isstring(x));
  p.addParameter('filler',defaults.filler,@(x) (ischar(x)||isstring(x)) &&numel(x)==1);
  p.addParameter('width',defaults.width,@(x) validateattributes(x,{'numeric'},{'>=' 20}));
  
  % parse inputs
  p.parse(varargin{:});
  
  filler = p.Results.filler;
  total_width = p.Results.width;
  logfile = p.Results.logfile;
  
  
  % construct line content
  title_width = numel(char(title));
  left_width = floor((total_width - title_width)/2);
  right_width = total_width - title_width - left_width;
  
  line_msg = strcat(repmat(filler,1,left_width),title,repmat(filler,1,right_width),'\n');
  
  Logging.print(line_msg,'logfile',logfile);
  
end