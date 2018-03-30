function batch_denoise(datapath,netpath,savepath)
% use every net in netpath to denoise every image in datapath, 
% and store them into corresponding folders
% 
% 

  net_files = dir(fullfile(netpath,'*mat'));
  [~,index] = sort({net_files.date});
  net_files = {net_files.name};
  net_files = net_files(index);
  
  [~,dataname] = fileparts(datapath);
  fprintf('Denoise %s images using net of each opoch...\n',dataname);
  
  
  load_net_fcn = @(path)Utilities.load_net(fullfile(netpath,path),'net');
  nets = cellfun(load_net_fcn,net_files,'UniformOutput',false);
  
  
  
  tmp_imds = cellfun(@(net)Utilities.denoise_datastore(net,datapath),nets,'UniformOutput',false);
  dest_imgs = cellfun(@(imds) imds.readall(),tmp_imds,'UniformOutput',false);
  
  
  % write images processed by nets of different epoches into the same folder
  for i = 1:length(dest_imgs{1})
    cur_path = fullfile(savepath,num2str(i));
    cur_prefix = fullfile(cur_path,'epoch_');
    mkdir(cur_path);
    for epoch = 1:length(dest_imgs)
      img_filename = strcat(cur_prefix,num2str(epoch),'.png');
      imwrite(dest_imgs{epoch}{i},img_filename);
    end
  end
end