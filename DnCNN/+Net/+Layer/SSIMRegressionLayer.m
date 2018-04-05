classdef SSIMRegressionLayer < nnet.layer.RegressionLayer
    % Regression Layer with SSIM loss function
    % 
    % Parameters:
    %   'Name'
    %   'Description'
    %   'Type'
    properties(SetAccess = private)
      sigma
    end
    
    properties(Access = private)
      w
    end
    
    properties(Constant)
      C1 = 1e-4;
      C2 = 3e-2^2;
    end
    
    methods
      function layer = SSIMRegressionLayer(varargin)
            
        p = inputParser();
        p.addOptional('sigma',5,@isnumeric);
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
        % Inputs (sequence):
        %   Y   Predictions made by network, of size:
        %   numResponses-by-numObservations-by-seqLength
        %   T   Targets (actual values), of size:
        %   numResponses-by-numObservations-by-seqLength
        
        layer.w = layer.get_gaussian_filter(size(Y,1));
        numElements = size(Y,4);
        ssim_vals = layer.cal_ssim(Y,T);
        loss = 1 - sum(ssim_vals) / numElements;
        loss = single(loss);
      end

      function dX = backwardLoss( layer, Y, T )
        
        layer.w = layer.get_gaussian_filter(size(Y,1));
        numElements = size(Y,4);
        dX = layer.cal_dX(Y,T) / numElements;
        dX = single(dX);
      end

    end
  
    methods(Access = private)
      
      function ssim_val = cal_ssim(self,Xpatch,Ypatch)
        % simple version of ssim
        numElements = size(Xpatch,4);
        ssim_val = gpuArray(zeros(numElements,1));
        for i = 1 : numElements
          cur_X = Xpatch(:,:,:,i);
          cur_Y = Ypatch(:,:,:,i);
          
          mux = self.w .* cur_X;
          mux = sum(mux(:));
          
          muy = self.w .* cur_Y;
          muy = sum(muy(:));
          
          sigmax2 = self.w .* (cur_X.^2) ;
          sigmax2 = sum(sigmax2(:)) - mux.^2;
          
          sigmay2 = self.w .* (cur_Y.^2);
          sigmay2 = sum(sigmay2(:)) - muy.^2;
          
          sigmaxy = self.w .* cur_X .*cur_Y;
          sigmaxy = sum(sigmaxy(:)) - mux.*muy;

          L = (2 .* mux .* muy + self.C1) ./(mux.^2 + muy.^2 + self.C1);
          CS = (2 .* sigmaxy + self.C2) ./ (sigmax2 + sigmay2 + self.C2);

          ssim_val(i) = L.*CS;
        end
      end
      
      function dX = cal_dX(self,Xpatch,Ypatch)
        numElements = size(Xpatch,4);
        dX = gpuArray(zeros(size(Xpatch)));
        for i = 1:numElements
          cur_X = Xpatch(:,:,:,i);
          cur_Y = Ypatch(:,:,:,i);
          
          mux = self.w .* cur_X;
          mux = sum(mux(:));
          
          muy = self.w .* cur_Y;
          muy = sum(muy(:));
          
          sigmax2 = self.w .* (cur_X.^2) ;
          sigmax2 = sum(sigmax2(:)) - mux.^2;
          
          sigmay2 = self.w .* (cur_Y.^2);
          sigmay2 = sum(sigmay2(:)) - muy.^2;
          
          sigmaxy = self.w .* cur_X .*cur_Y;
          sigmaxy = sum(sigmaxy(:)) - mux.*muy;
          
          L = (2 .* mux .* muy + self.C1) ./(mux.^2 + muy.^2 + self.C1);
          CS = (2 .* sigmaxy + self.C2) ./ (sigmax2 + sigmay2 + self.C2);

          dL = 2 .* self.w .* (muy - mux.*L) ./(mux.^2 + muy.^2 + self.C1);
          dCS = 2 ./ (sigmax2 + sigmay2 + self.C2) .* self.w .*...
            ((cur_Y - muy) - CS .* (cur_X-mux));

          dX(:,:,:,i) = - (dL.*CS + L.*dCS);
        end
      end
      
      function w = get_gaussian_filter(self,width)
        x = -(width/2):(width/2)-1;
        w = exp((-1 * x.^2)/(2 * self.sigma^2));
        w = w .* w';
        w = w/sum(w(:));
      end
    end

end
