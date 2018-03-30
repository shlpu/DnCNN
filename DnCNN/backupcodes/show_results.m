result_path = prj.path.results.Set12;
data_path = prj.path.test.Set12;
net_path = fullfile(prj.path.root,'net.mat');

net = Utilities.load_net(net_path,'net');

dirlist = dir(result_path);
dirnames = {dirlist.name};

for i = 3:length(dirnames)
    test_imds = imageDatastore(data_path,'ReadFcn',@(x)imnoise(im2single(imread(x)),'gaussian',0,(0.15/255)^2));
    img = test_imds.readimage(i);
    imshowpair(denoiseImage(img,net),img,'montage');
    pause(5);
end
