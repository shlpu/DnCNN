classdef SSIMRegressionLayer < nnet.layer.RegressionLayer
    % Regression Layer with SSIM loss function
    % 
    % Parameters:
    %   'Name' (parameter) : chararray | string (default '')
    %   'Description' (parameter) :  chararray | string (default 'SSIMRegressionLayer')
    %   'Type'  (parameter) :  chararray | string (default 'SSIMRegressionLayer')
    %   'sigma' (Optional) : numeric (default 1.5)
    %
    % Usage:
    %   reg_layer = SSIMRegressionLayer();
    %   reg_layer = SSIMRegressionLayer(Name,Value);
    %   reg_layer = SSIMRegressionLayer(sigma);
    %   reg_layer = SSIMRegressionLayer(sigma,Name,Value);
    
    
    properties(SetAccess = private)
      sigma
    end
    
    properties(Dependent)
      filtersize;
    end
    
    properties(Constant)
      enlarge_rate = 1000;
    end
    
    methods
      function filtersize = get.filtersize(self)
         filtersize = 2* ceil(self.sigma*3) + 1;
      end
    end
    
    methods
      function layer = SSIMRegressionLayer(varargin)
      % Return a regression Layer with SSIM loss function
      % 
      % Parameters:
      %   'Name' (parameter) : chararray | string (default '')
      %   'Description' (parameter) :  chararray | string (default 'SSIMRegressionLayer')
      %   'Type'  (parameter) :  chararray | string (default 'SSIMRegressionLayer')
      %   'sigma' (Optional) : numeric (default 1.5)
      %
      % Usage:
      %   reg_layer = SSIMRegressionLayer();
      %   reg_layer = SSIMRegressionLayer(Name,Value);
      %   reg_layer = SSIMRegressionLayer(sigma);
      %   reg_layer = SSIMRegressionLayer(sigma,Name,Value);    
      
        p = inputParser();
        p.addOptional('sigma',1.5,@isnumeric);
        p.addParameter('Name','',@isstring);
        p.addParameter('Description','SSIMRegressionLayer',@(x) ischar(x) || isstring(x));
        p.addParameter('Type','SSIMRegressionLayer',@(x) ischar(x) || isstring(x));
        p.parse(varargin{:});

        layer.sigma = p.Results.sigma;
        layer.Name = p.Results.Name;
        layer.Description = p.Results.Description;
        layer.Type = p.Results.Type;
      end

      function loss = forwardLoss( layer, Y, T )
        % forwardLoss    Return the SSIM loss between estimate
        % and true responses averaged by the number of observations
        %
        % Syntax:
        %   loss = layer.forwardLoss( Y, T );
        %
        % Inputs (image):
        %   Y   Predictions made by network, of size:
        %   height-by-width-by-numResponses-by-numObservations
        %   T   Targets (actual values), of size:
        %   height-by-width-by-numResponses-by-numObservations
        %
        
        numElements = size(Y,4);
        ssim_vals = 1 - layer.batch_cal_ssim(Y,T);
        ssim_vals = layer.enlarge_rate .* ssim_vals;
        loss = sum(ssim_vals) / numElements;
        loss = single(loss);
      end

      function dX = backwardLoss( layer, Y, T )
        
        numElements = size(Y,4);
        dX = layer.batch_cal_dX(Y,T);
        dX = layer.enlarge_rate .* dX;
        dX = single(dX) / numElements;
      end

    end
  
    methods(Access = private)
      
      function ssim_val = batch_cal_ssim(self,Xpatch,Ypatch)
        % simple version of ssim
        C = [1e-4 , 3e-2^2];
        w = get_gaussian_filter(self.sigma,self.filtersize);
        
        mux = imfilter(Xpatch,w,'replicate');
        muy = imfilter(Ypatch,w,'replicate');
        muxy = mux.*muy;
        mux2 = mux.^2;
        muy2 = muy.^2;

        sigmax2 = imfilter(Xpatch.^2,w,'replicate') - mux2;
        sigmay2 = imfilter(Ypatch.^2,w,'replicate') - muy2;
        sigmaxy = imfilter(Xpatch.*Ypatch,w,'replicate') - muxy;

        A1 = 2*mux.*muy+C(1);
        A2 = 2*sigmaxy+C(2);
        B1 = mux2+muy2+C(1);
        B2 = sigmax2+sigmay2+C(2);

        ssim_map = A1.*A2./(B1.*B2);

        ssim_val = mean(mean(mean(ssim_map,1),2),3);
        ssim_val = ssim_val(:);
      end
      
      function dX = batch_cal_dX(self,Xpatch,Ypatch)
        C = [1e-4 , 3e-2^2];
        n1 = self.filtersize(1); n2 = n1;
        delta = 1;
        [m,n,numChannels,~] = size(Xpatch);
        if numChannels ~=1
          error('Net:Layer:SSIMRegressionLayer','rgb images are not supported yet');
        end

        w = get_gaussian_filter(self.sigma,self.filtersize);

        
        X_patches = batch_image2patches(Xpatch, n1, n2, delta, delta);
        Y_patches = batch_image2patches(Ypatch, n1, n2, delta, delta);
        
        mux = imfilter(X_patches,w,'replicate');
        muy = imfilter(Y_patches,w,'replicate');
        muxy = mux.*muy;
        mux2 = mux.^2;
        muy2 = muy.^2;

        sigmax2 = imfilter(X_patches.^2,w,'replicate') - mux2;
        sigmay2 = imfilter(Y_patches.^2,w,'replicate') - muy2;
        sigmaxy = imfilter(X_patches.*Y_patches,w,'replicate') - muxy;

        A1 = 2*mux.*muy+C(1);
        A2 = 2*sigmaxy+C(2);
        B1 = mux2+muy2+C(1);
        B2 = sigmax2+sigmay2+C(2);

        factor = 2./(n1.^2.*B1.^2.*B2.^2);
        C1 = A1.*B1 .* (B2.* Y_patches - A2.* X_patches);
        C2 = B1 .*B2 .*(A2 -A1 ).*muy;
        C3 = A1 .*A2 .*(B1 -B2 ).*mux;
        gradientxSSIMxy = factor.*(C1+C2+C3);
        dX = - batch_patches2image(gradientxSSIMxy, m,n, delta, delta);
 
        if hasInfNaN(dX)
          warning('Net:Layer:SSIMRegressionLayer','Meet InfNAN error');
        end
      end
      
    end

end


function w = get_gaussian_filter(radius,filtersize)
  x = -(filtersize/2):(filtersize/2)-1;
  w = exp((-1 * x.^2)/(2 * radius^2));
  w = w .* w';
  w = w/sum(w(:));
end


function M_patches = batch_image2patches(im, n1, n2, delta1, delta2)
  if isa(im,"gpuArray")
    use_gpu = 1;
    im = gather(im);
  end
  [N1, N2,w,d] = size(im);

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
  M_patches = zeros(n1*n2, num_patches,w,d, 'single');
  for k = 1 : num_patches
    coor_x = XstartI(k):min(XstartI(k)+n1_minus1,N1);
    coor_y = YstartI(k):min(YstartI(k)+n2_minus1,N2);
    patch1 = 0*zeros(n1,n2,w,d);
    patch1(1:length(coor_x),1:length(coor_y),:,:) = im(coor_x,coor_y,:,:);
    M_patches(:, k,:,:) = reshape(patch1,[n1*n2,w,d]);
  end
  if use_gpu
    M_patches = gpuArray(M_patches);
  end

end


function im = batch_patches2image(M_patches, N1, N2, delta1, delta2)
  if isa(M_patches,"gpuArray")
    use_gpu = 1;
    M_patches = gather(M_patches);
  end
  [patch_size, num_patches,w,d] = size(M_patches);

  n1 = sqrt(patch_size);
  n2 = n1;


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

  im = zeros(N1, N2,w,d);
  M_weight = zeros(N1, N2);

  % use (one-layer) loop to extract the patches. This loop is inevitable in
  % the reconstruction phase (patches2image) because we need to add the
  % patches and accumulate the weight. 
  for k = 1 : num_patches
      cur_patch = reshape(M_patches(:,k,:,:),[n1, n2, w, d]);
      coor_x = XstartI(k):min(XstartI(k)+n1_minus1, N1);
      coor_y = YstartI(k):min(YstartI(k)+n2_minus1, N2);
      im(coor_x, coor_y , :, :) = im(coor_x, coor_y, :, :) + ...
        cur_patch(1:length(coor_x), 1:length(coor_y) , :, :);
      M_weight(coor_x, coor_y) = M_weight(coor_x, coor_y) + 1;
  end
  
  im = im./M_weight;
  if use_gpu
    im = gpuArray(im);
  end
end