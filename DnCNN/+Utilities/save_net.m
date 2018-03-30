function TF = save_net(savepath,net) %#ok<INUSD>lru
  try
    save(savepath,'net');
  catch
    TF = false;
    warning('Save net failed.');
    return;
  end
  TF = true;
end