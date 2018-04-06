reset(gpuDevice());
conv_size = [3,3];
net_width = 64;
num_color_channels = 1;
image_size = [50,50];
net_depth = 5;
num_res = (net_depth - 2)/3;

imds = imageDatastore(fullfile('Data','Train400'));
imds = denoisingImageDatastore(imds);

lgraph = Net.init_res_dncnn_network(image_size,'net_depth',net_depth);
plot(lgraph);

train_opts = trainingOptions('adam');

net = trainNetwork(imds,lgraph,train_opts);
