function print_info(info,varargin)
  % print information
  % 
  % Parameters:
  %   info (Required) : structure | class | cell
  %   logfile (Parameter) : string | char array
  %     specify the logging file


  if is_stringable(info)
    Logging.print(string(info),varargin{:});
    Logging.print('\n',varargin{:});
    return;
  end
  
  switch string(class(info))
    case "cell"
      print_cell(info,varargin{:});
    case "struct"
      print_structure(info,varargin{:});
    otherwise
      print_class(info,varargin{:});
  end
end

function print_class(info,varargin)
    names = properties(info);
    cellfun(@(name)print_property(info,name,varargin{:}),names);
end

function print_structure(info,varargin)

  names = fieldnames(info);
  cellfun(@(name)print_property(info,name,varargin{:}),names);
end

function print_cell(info,varargin)
  if ~all(is_stringable(info))
    error('MATLAB:invalidConversion',"This cell is not supported to be printed");
  end
  print_func = @(x) Logging.print(strcat(string(x),' '),varargin{:});
  cellfun(print_func,info);
end


function print_property(info, name, varargin)
  % print info.name, recursively if needed.
  % 
  % Parameters:
  %   info (Required) : structure | class | cell
  %   name (Required) : string
  %     field of info if it is structure
  %     readable properties if it is class
  %   logfile (Parameter) : string | char array
  %     specify the logging file
  
  if iscell(info)
    print_cell(info,varargin{:});
    return;
  end
  
  % read info.name property(field)
  eval_code = sprintf('cur_info = info.%s;',name);
  try
    eval(eval_code); % get cur_info
  catch
    return
  end
  
  
  % recursively deal structure
  if isstruct(cur_info) %#ok<NODEF>
    % cur_info might be a struct
    print_structure(cur_info,varargin{:});
    return;
  end
  
  
  % try to convert cur_info into string
  try
    cur_info = string(cur_info);
  catch e
    if strcmp(e.identifier,'MATLAB:invalidConversion')
      % if cur_info is not a class, an error will be throwed by print_class
      % from properties(cur_info)
      print_class(cur_info,varargin{:});
      return;
    end
  end
  
  % join cur_info together when it's a string array
  while numel(cur_info) ~= 1 
    cur_info = join(cur_info,'\t');
  end
  
  % print work goes here
  msg = sprintf("%s :\t%s\n",name, cur_info);
  Logging.print(msg,varargin{:});
end

function TF = is_stringable(x)
% check if a variable can be converted to string
  try
    string(x)
  catch
    TF = false;
    return;
  end
  TF = true;
end