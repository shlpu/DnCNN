classdef GaussianNoiseImageDatastore < Dataloader.noiseImageDatastore
%   A GaussianNoiseImageDatastore object encapsulates a datastore which
%   creates batches of Gaussian noisy image patches and corresponding noise patches
%   to be fed to a denoising deep neural network for training. Data Augmentation is used
%   if necessary.
% 
%   Differences from built-in denoisingImageDatastore:
%     1. support data augmentation
%
%   noiseImageDatastore properties:
%       PatchesPerImage         - Number of random patches to be extracted per image
%       PatchSize               - Size of the image patches
%       GaussianNoiseLevel      - Standard deviation of Gaussian noise
%       ChannelFormat           - Channel format of output noisy patches
%       MiniBatchSize           - Number of patches returned in each read
%       NumObservations         - Total number of patches in an epoch
%       DispatchInBackground    - Whether background dispatch is used
%       ImageAugmenter          - ImageDataAugmenter

  properties
    GaussianNoiseLevel % Standard deviation of Gaussian noise
  end
  
  methods(Access = public)
    function self = GaussianNoiseImageDatastore(imds,varargin)
      % construct a noiseImageDatastore
      %
      % Usage:
      %   ds = noiseImageDatastore(imds)
      %   ds = noiseImageDatastore(imds,Name,Value)
      % 
      % Parameters:
      %   * imds (Required)   : matlab.io.datastore.ImageDatastore
      %       imagedatastore instance passed to noiseImageDatastore
      %   * PatchesPerImage (Parameter)         : integer (default 512)
      %       specifying the number of pathces generated from an image.
      %   * PatchSize  (Parameter)              : integer | vector of two integers (default 50)
      %       specifying size of the random crops
      %   * GaussianNoiseLevel (Parameter)      : integer | vector of two integers (default 0.1)
      %       Standard deviation of Gaussian noise
      %       the vector of two integers [min_std,max_std] stands for the minimum and maximum of
      %       Gaussian noise level                         
      %   * ChannelFormat (Parameter)           : char array | string (defalut 'grayscale') 
      %       Specifies the data channel format as rgb or grayscale
      %   * DataAugmentation (Parameter)        : imageDataAugmenter object (default [])
      %
      %   Notes:
      %   -----
      %
      %  1. Training a deep neural network for a range of noise variances is a
      %      much more difficult problem compared to a single noise level one.
      %      Hence, it is recommended to create more patches compared to a single noise level
      %      case and training might take more time.
      %
      %  2. If channel format is grayscale, all color images would be converted to grayscale
      %     and if channel format is rgb, grayscale images would be replicated to
      %     simulate an rgb image.
      
      self = self@Dataloader.noiseImageDatastore(imds,varargin{:});
      
      p = inputParser();
      p.KeepUnmatched = true;
      p.addParameter('GaussianNoiseLevel',0.1,@validateGaussianNoiseLevel);
      p.parse(varargin{:});
      
      self.GaussianNoiseLevel = p.Results.GaussianNoiseLevel;
    end
  end
  
  methods(Access = protected)
    function [noisyPatch,residualNoise] = generateNoise(self,cleanPatch)
      isNoiseRange = (numel(self.GaussianNoiseLevel) == 2);
      
      if isNoiseRange
        noiseSigma = min(self.GaussianNoiseLevel) + ...
            abs(self.GaussianNoiseLevel(2)-self.GaussianNoiseLevel(1))*rand;
      else
        noiseSigma = self.GaussianNoiseLevel;
      end
      
      residualNoise = noiseSigma * randn(self.PatchSize,'single');
      noisyPatch = cleanPatch + residualNoise;
    end
  end
  
  
end

function B = validateGaussianNoiseLevel(GaussianNoiseLevel)

supportedClasses = {'single','double'};
attributes = {'nonempty','real','vector', ...
    'nonnegative','finite','nonsparse','nonnan','nonzero','>=',0,'<=',1};

validateattributes(GaussianNoiseLevel, supportedClasses, attributes,...
    mfilename,'GaussianNoiseLevel');

if numel(GaussianNoiseLevel) > 2
    error(message('Dataloader:GaussianNoiseImageDatastore:invalidGaussianNoiseLevel'));
end

B = true;

end