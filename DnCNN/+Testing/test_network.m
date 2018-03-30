function test_network(prj)
  % test performance of network
  %  performance on:
  %   1. PSNR and SSIM
  %   2. denoising visual quality
  %   3. processing speed
  %
  %
  % Parameters:
  %   prj (Required) : structure
  %
  
  logfile = prj.path.log.performance;
  
  % delete previous testfiles
  delete_testfile(prj.path);

  Logging.print_line("BEGIN TESTING",'logfile',logfile,'filler','=');
  Logging.print_line(string(datetime),'logfile',logfile,'filler','=');
  
  Logging.print_project_info(prj,'logfile',logfile);
  
  % test psnr and ssim of all dataset
  Testing.test_psnr_ssim(prj.path.Checkpoint, prj.path.test.BSD68, prj.noise, 'logfile',logfile);
  Testing.test_psnr_ssim(prj.path.Checkpoint, prj.path.test.Set12, prj.noise, 'logfile',logfile);
  
  % test specific dataset and save denoised images
  structfun(@(x)Utilities.rebuild_dir(x),prj.path.results);
  Testing.batch_denoise(prj.path.test.Set12, prj.path.Checkpoint, prj.path.results.Set12);
%   Testing.batch_denoise(prj.path.test.Set12, prj.path.Checkpoint, prj.path.results.Set68);
  
  % speed test
  net = Utilities.load_net(prj.path.Checkpoint);
  Testing.test_speed(net,'logfile',logfile);
  
  % end testing
  Logging.print('\n\n','logfile',logfile);
  Logging.print_line(string(datetime),'logfile',logfile,'filler','=');
  Logging.print_line("END TESTING",'logfile',logfile,'filler','=');
end


function delete_testfile(pathinfo)
% delete previous testfile

  delete(fullfile(pathinfo.root,'PSNR*.tif'));
  delete(fullfile(pathinfo.root,'performance.*'));
  structfun(@(x)Utilities.rebuild_dir(x),pathinfo.results);
  
end


