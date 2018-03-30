function test_psnr_ssim(checkpointpath,datapath,noiseinfo,varargin)
% Test PSNR and SSIM of each network in checkpointpath
  
  % set path
  rootpath = fileparts(checkpointpath);
  figprefix.PSNR = fullfile(rootpath,'PSNR_');
  figprefix.SSIM = fullfile(rootpath,'SSIM_');
  [~,dataname] = fileparts(datapath);
  
  % test pipline
  [src_psnr,src_ssim] = batch_cal_psnr_ssim(checkpointpath,datapath,noiseinfo);
  Logging.print_line(dataname,varargin{:});
  print_psnr_ssim(src_psnr,src_ssim,varargin{:});
  
  % plot and save figures
  fig = figure;
  plot(src_psnr);
  print(strcat(figprefix.PSNR,dataname),'-dtiffn');
  plot(src_ssim);
  print(strcat(figprefix.SSIM,dataname),'-dtiffn');
  close(fig);
end

function [test_psnr,test_ssim] = batch_cal_psnr_ssim(checkpointpath,datapath,noiseinfo)
% calculate PSNR and SSIM of images 
  

  % load test data
  [~,dataname] = fileparts(datapath);
  imds = imageDatastore(datapath,...
    'IncludeSubfolders',true,...
    'FileExtensions',{'.jpg','.png','.bmp'});
  ref_imgs = imds.readall();
  ref_imgs = cellfun(@im2double,ref_imgs,'UniformOutput',false);
  input_imgs = cellfun(@(x)imnoise(x,'gaussian',0,(noiseinfo.std/255)^2),...
                          ref_imgs,'UniformOutput',false);
  
  % read net files
  net_files = dir(fullfile(checkpointpath,'*mat'));
  [~,index] = sort({net_files.date});
  net_files = {net_files.name};
  net_files = net_files(index);
  
  % test for each epoch
  test_psnr = zeros(numel(index),1);
  test_ssim = zeros(numel(index),1);
  fprintf('Calculating %s PSNR and SSIM of each epoch(%d):\n',dataname,numel(index));
  for i = 1:numel(index)
    
    net = Utilities.load_net(fullfile(checkpointpath,net_files{i}));
    [~,test_psnr(i)] = Utilities.cal_psnr(net,ref_imgs,input_imgs);
    [~,test_ssim(i)] = Utilities.cal_ssim(net,ref_imgs,input_imgs);
    
    fprintf('.')
  end
  fprintf('\n')
  
end

function print_psnr_ssim(src_psnr,src_ssim,varargin)


  Logging.print("Epoch\t\tPSNR(AVR)\t\tSSIM(AVR)\n",varargin{:});
  
  for i = 1:numel(src_psnr)
    msg = sprintf("%d\t\t%.3f\t\t\t%.3f\n",i,src_psnr(i),src_ssim(i));
    Logging.print(msg,varargin{:});
  end
  
  Logging.print_line('',varargin{:});
  Logging.print('\n\n',varargin{:});
  
  % get largest PSNR and SSIM
  [max_psnr,i] = max(src_psnr);
  msg = sprintf("Max PSNR %.3f at epoch %d (total %d)\n",max_psnr,i,numel(src_psnr));
  Logging.print(msg,varargin{:});
  [max_ssim,i] = max(src_ssim);
  msg = sprintf("Max SSIM %.3f at epoch %d (total %d)\n",max_ssim,i,numel(src_psnr));
  Logging.print(msg,varargin{:});
end