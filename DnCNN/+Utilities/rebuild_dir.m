function rebuild_dir(path)
  % delete and rebuild the dir
  if exist(path,'dir')
    rmdir(path,'s');
  end
  mkdir(path);
end