function [net_psnr,mean_psnr] = cal_psnr(net,ref_imgs,input_imgs)
% calculate Peak Signal-to-Noise Ratio(PSNR) of the network
% 
% Return:
%   net_psnr    : cell
%     PSNR of each images in noisy_imds
%   mean_psnr   : numeric
%     mean PSNR
% 
% Parameters:
%   net        (Required)  : Serial network
%     network for testing
%   ref_imgs   (Required)  : cell
%     Reference images
%   input_imgs (Required)  : cell
%     input images for net
%
% Note:
%   1. each image of ref_imgs and input_imgs should be corresponded
  
  func = @(input,clean) psnr(Utilities.denoiseImage(input,net),clean);
  net_psnr = cellfun(func,input_imgs,ref_imgs);
  mean_psnr = mean(net_psnr);
  
end
