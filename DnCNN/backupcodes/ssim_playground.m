% clear all;

ref_img = im2single(imread('Data/Train400/test_001.png'));
ref_img = gpuArray(ref_img);
ref_img = imresize(ref_img,1);
img = imnoise(ref_img,'gaussian');
% real_ssim = ssim(img,ref_img);

self_ssim = cal_ssim(img,ref_img);

dX = cal_dX(img,ref_img);

function dX = cal_dX(Xpatch,Ypatch)
  filtersize = [11,11];
  sigma = 1.5;
  C = [1e-4 , 3e-2^2];
  n1 =  filtersize(1); n2 = n1; delta1 = 5; delta2 = 5;
  [m,n,numChannels,numElements] = size(Xpatch);

  w = get_gaussian_filter(sigma,filtersize);

  dX = gpuArray(zeros(size(Xpatch)));
  for i = 1:numElements
    cur_X = Xpatch(:,:,:,i);
    cur_Y = Ypatch(:,:,:,i);

    mux = imfilter(cur_X,w,'replicate');
    muy = imfilter(cur_Y,w,'replicate');
    muxy = mux.*muy;
    mux2 = mux.^2;
    muy2 = muy.^2;

    sigmax2 = imfilter(cur_X.^2,w,'replicate') - mux2;
    sigmay2 = imfilter(cur_Y.^2,w,'replicate') - muy2;
    sigmaxy = imfilter(cur_X.*cur_Y,w,'replicate') - muxy;

    A1 = 2*mux.*muy+C(1);
    A2 = 2*sigmaxy+C(2);
    B1 = mux2+muy2+C(1);
    B2 = sigmax2+sigmay2+C(2);

    A_patches = image2patches(cur_X, n1, n2, delta1, delta2);
    Ref_patches = image2patches(cur_Y, n1, n2, delta1, delta2);
    num_patches = size(A_patches,2);
    
    B1 = B1(1:num_patches);
    B2 = B2(1:num_patches);
    A1 = A1(1:num_patches);
    A2 = A2(1:num_patches);
    mux = mux(1:num_patches);
    muy = muy(1:num_patches);
    
    factor = 2./(n1.^2.*B1.^2.*B2.^2);
    C1 = A1.*B1 .* ( B2.*Ref_patches - A2.* A_patches);
    C2 = B1 .*B2 .*(A2 -A1 ).*muy;
    C3 = A1 .*A2 .*(B1 -B2 ).*mux;
    gradientxSSIMxy = factor.*(C1+C2+C3);
    

    dX(:,:,:,i) =  patches2image(gradientxSSIMxy, m,n, delta1, delta2);
  end

  dX = -dX;
end



function ssim_val = cal_ssim(img,ref_img)
  w = get_gaussian_filter(1.5,11);
  
  C = [1e-4 , 3e-2^2];
  
  mux = imfilter(img,w,'replicate');
  muy = imfilter(ref_img,w,'replicate');
  muxy = mux.*muy;
  mux2 = mux.^2;
  muy2 = muy.^2;

  sigmax2 = imfilter(img.^2,w,'replicate') - mux2;
  sigmay2 = imfilter(ref_img.^2,w,'replicate') - muy2;
  sigmaxy = imfilter(img.*ref_img,w,'replicate') - muxy;

  A1 = 2*mux.*muy+C(1);
  A2 = 2*sigmaxy+C(2);
  B1 = mux2+muy2+C(1);
  B2 = sigmax2+sigmay2+C(2);

  ssim_map(:,:) = A1.*A2./(B1.*B2);
  ssim_val = mean(ssim_map(:));
end



function w = get_gaussian_filter(sigma,width)
  x = -(width/2):(width/2)-1;
  w = exp((-1 * x.^2)/(2 * sigma^2));
  w = w .* w';
  w = w/sum(w(:));
end


function M_patches = image2patches(im, n1, n2, delta1, delta2)
% Transfer an image to patches of size n1 x n2. The patches are sampled
% from the images with a translating distance delta1 x delta2. The patch sampling
% starts from (i1, j1) and ends at (i2, j2).
%
%
% Guoshen Yu, 2009.09.30
  
  [N1, N2] = size(im);

  % the coordinates of the top-left point in all the patches are computed and
  % stored in (XstartI, YstartI). XstartI or YstartI is a vector of length
  % #patches.
  Xstart = uint16(1 : delta1 : N1 - n1 + delta1 + 1);
  Ystart = uint16(1 : delta2 : N2 - n2 + delta2 + 1);
  [YstartI, XstartI] = meshgrid(Ystart, Xstart);
  YstartI = YstartI(:);
  XstartI = XstartI(:);

  n1_minus1 = n1 - 1;
  n2_minus1 = n2 - 1;

  % use (one-layer) loop to extract the patches. This loop is inevitable in
  % the reconstruction phase (patches2image) because we need to add the
  % patches and accumulate the weight.
  num_patches = length(XstartI);
  M_patches = zeros(n1*n2, num_patches, 'single');
  if isa(im,"gpuArray")
    M_patches = gpuArray(M_patches);
    for k = 1 : num_patches
      coor_x = XstartI(k):min(XstartI(k)+n1_minus1,N1);
      coor_y = YstartI(k):min(YstartI(k)+n2_minus1,N2);
      patch1 = gpuArray(0*zeros(n1,n2));
      patch1(1:length(coor_x),1:length(coor_y)) = im(coor_x,coor_y);
      M_patches(:, k) = patch1(:);
    end
  else
    for k = 1 : num_patches
      coor_x = XstartI(k):min(XstartI(k)+n1_minus1,N1);
      coor_y = YstartI(k):min(YstartI(k)+n2_minus1,N2);
      patch1 = 0*zeros(n1,n2);
      patch1(1:length(coor_x),1:length(coor_y)) = im(coor_x,coor_y);
      M_patches(:, k) = patch1(:);
    end
  end
end

function im = patches2image(M_patches, N1, N2, delta1, delta2)
% Reput patches in images in the appropriate positions with appropriate
% weight. It is the inverse procedure of image2patches_fast
% 
% Guoshen Yu, 2009.10.23

[patch_size, num_patches] = size(M_patches);

n1 = sqrt(patch_size);
n2 = n1;

% % % n1 = 110;
% % % n2 = 1;

% the coordinates of the top-left point in all the patches are computed and
% stored in (XstartI, YstartI). XstartI or YstartI is a vector of length
% #patches. 
  Xstart = uint16(1 : delta1 : N1 - n1 + delta1 + 1);
  Ystart = uint16(1 : delta2 : N2 - n2 + delta2 + 1);
  [YstartI, XstartI] = meshgrid(Ystart, Xstart);
  YstartI = YstartI(:);
  XstartI = XstartI(:);

  n1_minus1 = n1 - 1;
  n2_minus1 = n2 - 1;

  if isa(M_patches,"gpuArray")
    im = gpuArray(zeros(N1, N2));
  end
  M_weight = zeros(N1, N2);

  % use (one-layer) loop to extract the patches. This loop is inevitable in
  % the reconstruction phase (patches2image) because we need to add the
  % patches and accumulate the weight. 
  for k = 1 : num_patches
      cur_patch = reshape(M_patches(:,k),[n1, n2]);
      coor_x = XstartI(k):min(XstartI(k)+n1_minus1, N1);
      coor_y = YstartI(k):min(YstartI(k)+n2_minus1, N2);
      im(coor_x, coor_y) = im(coor_x, coor_y) + cur_patch(1:length(coor_x), 1:length(coor_y));
      M_weight(coor_x, coor_y) = M_weight(coor_x, coor_y) + 1;
  end
  
  im = im./M_weight;

end