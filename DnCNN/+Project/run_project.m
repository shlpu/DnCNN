function net = run_project(opts,varargin)
  % project pipline
  opts = Project.project_generator(opts,varargin{:});
  logfile = opts.path.log.training;
  start_time = datetime;
  Logging.print_line(string(start_time),'logfile',logfile);
	Logging.print_project_info(opts,'logfile',logfile);
  
  try
    net = Training.train_network(opts);
  catch
    Logging.log_errors(opts,'logfile','error.txt');
    net = [];
    return;
  end
  Utilities.save_net(fullfile(opts.path.root,'net.mat'),net);
  
  
  time_used = datetime - start_time;
  Logging.print_line(string(start_time),'logfile',logfile);
	Logging.print_project_info(opts,'logfile',logfile);
  Logging.print_line(strcat('Time spend: ',string(time_used)),'logfile',logfile);
  
  try
    Testing.test_network(opts);
  catch
    Logging.log_errors(opts,'logfile','error.txt');
    net = [];
    return;
  end
end
