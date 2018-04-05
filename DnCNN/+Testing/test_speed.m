function speeds = test_speed(net,varargin)
% test speed of denoising network
  imgsizes = [128,256,512,1024];
  imgnum = 100;
  
  fprintf("Testing net processing speed:\n");
  speeds = zeros(size(imgsizes));
  for i = 1:numel(imgsizes)
    tic;
    fprintf("%d...",imgsizes(i));
    for j = 1:imgnum
      img = randn([imgsizes(i),imgsizes(i),net.Layers(1).InputSize(3)]);
      Utilities.denoiseImage(img,net);
    end
    time_used = toc;
    speeds(i) = imgnum/time_used;
  end 
  fprintf("\n");
  
  Logging.print("Approximate FPS (images/seconds)\n",varargin{:})
  speed_log_fcn = @(imgsize,speed) Logging.print(sprintf("size %d : %.3f\n",imgsize,speed),varargin{:});
  arrayfun(speed_log_fcn,imgsizes,speeds);
end
