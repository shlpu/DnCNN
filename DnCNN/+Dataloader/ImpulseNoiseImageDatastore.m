classdef ImpulseNoiseImageDatastore < Dataloader.noiseImageDatastore
%   A GaussianNoiseImageDatastore object encapsulates a datastore which
%   creates batches of Impusive noisy image patches and corresponding noise patches
%   to be fed to a denoising deep neural network for training. Data Augmentation is used
%   if necessary.
%
%   Differences from built-in denoisingImageDatastore:
%     1. support data augmentation
%     2. add impulse noise to image patches
%     
%   noiseImageDatastore properties:
%       PatchesPerImage         - Number of random patches to be extracted per image
%       PatchSize               - Size of the image patches
%       NoiseDensity            - Impulse noise density
%       NoiseValueRange         - Range of value of noise
%       ChannelFormat           - Channel format of output noisy patches
%       MiniBatchSize           - Number of patches returned in each read
%       NumObservations         - Total number of patches in an epoch
%       DispatchInBackground    - Whether background dispatch is used

  properties
    NoiseDensity
    NoiseValueRange
  end
  
  methods(Access = public)
    function self = ImpulseNoiseImageDatastore(imds,varargin)
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
      %   * NoiseDensity (Parameter)            : numeric value in range[0,1] (default 0.05)
      %       specifying the density of impulse noise
      %   * NoiseValueRange (Parameter)         : numeric array     (default [0,1])     
      %       specifying the accepted value of impulse noise, by default it is salt&pepper noise
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
      p.addParameter('NoiseDensity',0.05,@validateNoiseDensity);
      p.addParameter('NoiseValueRange',[0,1],@validateNoiseValueRange);
      p.parse(varargin{:});
      
      self.NoiseDensity = p.Results.NoiseDensity;
      self.NoiseValueRange = p.Results.NoiseValueRange;
    end
  end
  
  methods(Access = protected)
    function [noisyPatch,impulsiveNoise] = generateNoise(self,cleanPatch)
      [w,d,~] = size(cleanPatch);
      
      % make mask: -1 for uncontaminated pixel, 1 for contaminated pixel
      mask = randsrc(w,d,[-1,1;1-self.NoiseDensity,self.NoiseDensity]);
      
      % make uniform random value mask based on NoiseValueRange
      num_value = numel(self.NoiseValueRange);
      index = randi(num_value,w,d);
      impulsiveNoise = zeros(w,d);
      for i = 1:num_value
        impulsiveNoise(index==i) = self.NoiseValueRange(i);
      end
      impulsiveNoise(mask == -1) = -1;
      
      % add impulse noise
      noisyPatch = cleanPatch;
      noisyPatch(mask==1) = impulsiveNoise(mask==1);
      
    end
  end
  
  
end

function B = validateNoiseDensity(NoiseDensity)

supportedClasses = {'single','double'};
attributes = {'nonempty','real', ...
    'nonnegative','finite','nonsparse','nonnan','nonzero','>=',0,'<=',1};

validateattributes(NoiseDensity, supportedClasses, attributes,...
    mfilename,'NoiseDensity');

if numel(NoiseDensity) > 1
    error(message('Dataloader:ImpulseNoiseImageDatastore:invalidNoiseDensity'));
end

B = true;

end

function B = validateNoiseValueRange(NoiseValueRange)

supportedClasses = {'single','double'};
attributes = {'nonempty','real', ...
    'nonnegative','finite','nonsparse','nonnan','>=',0,'<=',1};

validateattributes(NoiseValueRange, supportedClasses, attributes,...
    mfilename,'NoiseDensity');

if all(size(NoiseValueRange)~=1)
  error(message('Dataloader:ImpulseNoiseImageDatastore:invalidNoiseDensity'));
end
B = true;

end