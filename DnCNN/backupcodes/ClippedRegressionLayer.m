classdef ClippedRegressionLayer < nnet.layer.RegressionLayer
    % Regression Layer with gradient clipping
    % 
    % Parameters:
    %   'Threshold'
    %   'Name'
    %   'Description'
    %   'Type'
    properties
        threshold;
    end
 
    methods
      function layer = ClippedRegressionLayer(varargin)
            
        p = inputParser();
        p.addOptional('Threshold',0.005);
        p.addOptional('Name','',@isstring);
        p.addOptional('Description','ClippedRegressionLayer',@isstring);
        p.addOptional('Type','ClippedRegressionLayer',@isstring);
        p.parse(varargin{:});


        layer.Name = p.Results.Name;
        layer.Description = p.Results.Description;
        layer.Type = p.Results.Type;
        layer.threshold = p.Results.Threshold;
      end

      function loss = forwardLoss( layer, Y, T )
        % forwardLoss    Return the MSE loss between estimate
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

        squares = 0.5*(Y-T).^2;
        numElements = size(Y,4);
        loss = sum( squares (:) ) / numElements;
      end

      function dX = backwardLoss( layer, Y, T )


        dX = (Y - T)./size(Y,4);
        dX = ClippedRegressionLayer.clip_gradient(dX,layer.threshold);
      end

  end

  methods(Static,Access=private)
    
    function gradient = clip_gradient(gradient,threshold)
      % Mini-batch Gradient Clipping
      % This method is sugguested by Mikolov, see arxiv:1211.5063
      for i = 1:size(gradient,4)
        gradient(:,:,:,i) = single_clip(gradient(:,:,:,i),threshold);
      end
      
      function gradient = single_clip(gradient,threshold)
        % Gradient Clipping for each image of the mini-batch
        if length(size(gradient)) == 2
          loop_length = 1;
        else
          loop_length = size(gradient,3);
        end
        for j = 1:loop_length
          energy = norm(gradient(:,:,j));
          if threshold ~= 0 && energy > threshold
            gradient(:,:,j) = gradient(:,:,j) .* threshold/energy;
          end
        end
      end
      
    end
    
  end
end