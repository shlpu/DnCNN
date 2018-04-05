function [net_ssim, mean_ssim] = cal_ssim(net,ref_imgs,input_imgs)
% calculate Structural Similarity Index (SSIM) of the network
% 
% Return:
%   net_ssim    : cell
%     SSIM of each images in noisy_imds
%   mean_ssim   : numeric
%     mean SSIM
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
  
  func = @(input,clean) ssim(Utilities.denoiseImage(input,net),clean);
  net_ssim = cellfun(func,input_imgs,ref_imgs);
  mean_ssim = mean(net_ssim);
  
end
