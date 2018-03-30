function stop = denoising_training_logger(info,N,cal_psnr,performance_log,training_log)
% denoising_training_logger do the following things:
%   1. write training informations into training logfile
%   2. calculate the PSNR and SSIM of the network at the begining of each epoch,
%       and write into the performance logfile
%   3. stop training if PSNR and SSIM converges, i.e., not improving for more than N epochs.
%
%   Return:
%     stop  : logical
%       true to stop training progress, false to continue training
%
%   Parameters:
%     info  : structure
%       information passed by TrainNetwork
%     N     : integer
%       if PSNR hasn't imporved for N epochs, then stop training
%     cal_psnr

stop = false;

% Keep track of the best validation accuracy and the number of validations for which
% there has not been an improvement of the accuracy.
persistent bestPSNR
persistent valLag
persistent lastepoch

% Clear the variables when training starts.
if info.Epoch == 1
    bestPSNR = 0;
    valLag = 0;
    lastepoch = 1;
    
elseif info.Epoch == lastepoch + 1
    
    [cur_psnr,~] = mean(cal_psnr());
    
    if cur_psnr > bestPSNR
        valLag = 0;
        bestPSNR = cur_psnr;
    else
        valLag = valLag + 1;
    end
    
    % If the validation lag is at least N, that is, the psnr
    % has not improved for at least N validations, then return true and
    % stop training.
    if valLag >= N
        stop = true;
    end
    
end

end



