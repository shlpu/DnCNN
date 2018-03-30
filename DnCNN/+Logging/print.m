function print(msg,varargin)
  % print msg to console and logfile(if provided), a simple wrapper of fprintf function
  % 
  % Parameters:
  %   msg:  (Required) char array | string
  %     message to print
  %   logfile: (Parameter) char array | string
  %     log file path
  %     example: "log.txt"
  %   
  % Usage:
  %   printf(msg); % print msg to consolo
  %   printf(msg,filename); % append msg to end of logfile
  % 
  % Example:
  %   msg = sprintf('%d\t%d\n',1,2);
  %   Logging.print(msg,'logfile',logfile);
  
  p = inputParser();
  p.addParameter('logfile',"CONSOLE",@(x) ischar(x)||isstring(x));
  p.parse(varargin{:});
  
  msg = string(msg);
  
  % join msg together when it's a string array
  while numel(msg) ~= 1 
    msg = join(msg,'\t');
  end
  
  if p.Results.logfile == "CONSOLE"
    fprintf(msg);
  else
    fID = fopen(p.Results.logfile,'a');
    fprintf(fID,msg);
    fprintf(msg);
    fclose(fID);
  end
end