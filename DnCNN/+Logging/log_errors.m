function log_errors(opts)
% called when an error is thrown during the project
% log error and project information for debug
% 
% Parameters:
%   opts  (Required) : structure
%     project configurations and information
%
% Example 1:
%   try
%     net = Training.train_network(opts);
%   catch
%     Logging.log_errors(opts);
%     net = [];
%     return;
%   end
% 
%
    logfile = 'error.txt';
    
    % remove irrelevant information  
    opts = rmfield(opts,'path');
    opts = rmfield(opts,'prefix');
    opts.imds = rmfield(opts.imds,'augmenter');
    opts.train = rmfield(opts.train,'Verbose');
    opts.train = rmfield(opts.train,'VerboseFrequency');
    opts.train = rmfield(opts.train,'Plots');
    
    % format training options as readable json 
    info_msg = jsonencode(opts);
    info_msg = replace(info_msg,',',',\n');
    info_msg = replace(info_msg,'{','{\n');
    info_msg = replace(info_msg,'}','\n}');
    
    err_msg = string(lasterr); %#ok<LERR>
    
    
    Logging.print_line(string(datetime),'logfile',logfile,'filler','=');
    
    Logging.print(err_msg,'logfile',logfile);
    Logging.print('\n\n\n','logfile',logfile);
    
    Logging.print_line('Training Options','logfile',logfile);
    Logging.print(info_msg,'logfile',logfile);
    Logging.print('\n','logfile',logfile);
    
    Logging.print_line('','logfile',logfile);
    Logging.print_line('','logfile',logfile,'filler','=');
    Logging.print('\n','logfile',logfile);
    
end