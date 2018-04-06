%noiseImageDatastore   enhanced denoisingImageDatastore 
%
%   This is an enhanced implementation of built-in denoisingImageDatastore class.
%
%   Differences from built-in denoisingImageDatastore:
%     1. support data augmentation
%
%   noiseImageDatastore properties:
%       PatchesPerImage         - Number of random patches to be extracted per image
%       PatchSize               - Size of the image patches
%       ChannelFormat           - Channel format of output noisy patches
%       MiniBatchSize           - Number of patches returned in each read
%       NumObservations         - Total number of patches in an epoch
%       DispatchInBackground    - Whether background dispatch is used
%       ImageAugmenter          - ImageDataAugmenter
%
%   noiseImageDatastore methods:
%       noiseImageDatastore     - Construct a noiseImageDatastore
%       hasdata                 - Returns true if there is more data in the datastore
%       partitionByIndex        - Partitions a noiseImageDatastore given indices
%       preview                 - Reads the first image from the datastore
%       read                    - Reads a MiniBatch of data from the datastore
%       readall                 - Reads all observations from the datastore
%       readByIndex             - Random access read from datastore given indices
%       reset                   - Resets datastore to the start of the data
%       shuffle                 - Shuffles the observations in the datastore

classdef (Abstract) noiseImageDatastore < ...
        matlab.io.Datastore &...
        matlab.io.datastore.MiniBatchable &...
        matlab.io.datastore.Shuffleable & ...
        matlab.io.datastore.BackgroundDispatchable &...
        matlab.io.datastore.PartitionableByIndex

    properties (SetAccess = private, GetAccess = public)     
      PatchesPerImage % Number of random patches to be extracted per image
      PatchSize % Size of the image patches
      ChannelFormat % Channel format of output noisy patches
      ImageAugmenter % data augmenter
    end
    
    properties(Dependent)
      MiniBatchSize % Number of patches returned in each read
    end

    properties(SetAccess = protected, Dependent)
      NumObservations
    end

    properties (Access = private, Hidden, Dependent)
      TotalNumberOfMiniBatches
    end

    properties (Access = private)
      imds % internal datastore
    end

    properties (Access = private)        
      CurrentMiniBatchIndex
      NumberOfChannels
      MiniBatchSizeInternal
      OrderedIndices
    end
        
    methods
        
      function batchSize = get.MiniBatchSize(self)
        batchSize = self.MiniBatchSizeInternal;
      end
        
      function set.MiniBatchSize(self, batchSize)
        self.MiniBatchSizeInternal = batchSize;
      end
        
      function tnmb = get.TotalNumberOfMiniBatches(self)

        tnmb = floor(self.NumObservations/self.MiniBatchSize) + ...
            (mod(self.NumObservations, self.MiniBatchSize) > 0)*1;

      end
        
      function numObs = get.NumObservations(self)
        numObs = length(self.OrderedIndices);
      end
             
      function self = noiseImageDatastore(imds,varargin)
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
      %                               
      %                               
      %   
      %   

        images.internal.requiresNeuralNetworkToolbox(mfilename);


        validateImagedatastore(imds);
        options = parseInputs(varargin{:});

        self.PatchesPerImage = options.PatchesPerImage;
        self.ChannelFormat = options.ChannelFormat;
        if strcmp(self.ChannelFormat,'rgb')
          self.NumberOfChannels = 3;
        else
          self.NumberOfChannels = 1;
        end
        if numel(options.PatchSize) == 1
          self.PatchSize = [options.PatchSize options.PatchSize self.NumberOfChannels];
        else
          self.PatchSize = [options.PatchSize self.NumberOfChannels];
        end


        self.imds = imds.copy(); % Don't mess with state of imds input
        self.ImageAugmenter = options.DataAugmentation;
        
        self.DispatchInBackground = options.DispatchInBackground;
        self.MiniBatchSize = 128;
        numObservations = length(self.imds.Files) * self.PatchesPerImage;
        self.OrderedIndices = 1:numObservations;

        self.reset();
      end
        
    end
    
     methods(Static, Hidden = true)
        function self = loadobj(S)
            self = denoisingImageDatastore(S.imds, ...
                'ChannelFormat', S.ChannelFormat, ...
                'DataAugmentation',S.ImageAugmenter,...
                'PatchesPerImage', S.PatchesPerImage,...
                'PatchSize', [S.PatchSize(1) S.PatchSize(2)], ...
                'BackgroundExecution', S.BackgroundExecution);
        end
    end
    
    methods (Hidden)
        function S = saveobj(self)
            % Serialize denoisingImageDatastore object
            % Note we that serialize DispatchInBackground under the name
            % BackgroundExecution to make V1 and V2 loadobj work.
            S = struct('imds',self.imds,...
                'ChannelFormat',self.ChannelFormat,...
                'DataAugmentation',self.ImageAugmenter,...
                'PatchesPerImage',self.PatchesPerImage,...
                'PatchSize',self.PatchSize,...
                'BackgroundExecution',self.DispatchInBackground);            
        end
        
    end
    
    methods

      function [data,info] = readByIndex(self,indices)
        % hierated from matlab.io.datastore.Shuffleable
        indices = self.OrderedIndices(indices);

        startMod = mod(indices(1), self.PatchesPerImage);
        endMod = mod(indices(end), self.PatchesPerImage);

        isStartModNonZero = (startMod > 0);
        isEndModNonZero = (endMod > 0);
        startImage = floor(indices(1)/self.PatchesPerImage) + isStartModNonZero*1;
        endImage = floor(indices(end)/self.PatchesPerImage) + isEndModNonZero*1;

        % Create datastore partition via a copy and index. This is
        % faster than constructing a new datastore with the new
        % files.
        subds = copy(self.imds);
        subds.Files = self.imds.Files(startImage:endImage);
        images = subds.readall();

        if startImage == endImage
            [input,response] = self.getNoisyPatches(images, length(indices));
        else
            startImageNumPatches = isStartModNonZero * ...
                (self.PatchesPerImage - startMod) + 1;
            endImageNumPatches = (~isEndModNonZero) * ...
                self.PatchesPerImage + endMod;

            numPatches = [startImageNumPatches ...
                self.PatchesPerImage*ones(1,endImage-startImage-1)...
                endImageNumPatches];

            [input,response] = self.getNoisyPatches(images, numPatches);
        end
        data = table(input,response);
        info.CurrentFileIndices = startImage:endImage;
      end

      function [data,info] = read(self)
        % hierated from matlab.io.Datastore
        if ~self.hasdata()
           error(message('images:noiseImageDatastore:outOfData')); 
        end

        batchNumber = self.CurrentMiniBatchIndex;
        startObsIndex = (batchNumber - 1) * self.MiniBatchSize + 1;
        if batchNumber == self.TotalNumberOfMiniBatches
            endObsIndex = self.NumObservations;
        else
            endObsIndex = startObsIndex + self.MiniBatchSize - 1;
        end

        self.CurrentMiniBatchIndex = self.CurrentMiniBatchIndex + 1;
        [data,info] = self.readByIndex(startObsIndex:endObsIndex);
      end

      function reset(self)
        % hierated from matlab.io.Datastore
        self.imds.reset();
        self.CurrentMiniBatchIndex = 1;
      end

      function newds = shuffle(self)
        % hierated from matlab.io.datastore.Shuffleable
        newds = copy(self);
        imdsIndexList = randperm(length(self.imds.Files));
        reorderIndexList(newds,imdsIndexList);
      end

      function TF = hasdata(self)
        % hierated from matlab.io.Datastore
       TF = self.CurrentMiniBatchIndex <= self.TotalNumberOfMiniBatches;
      end
      
      function newds = partitionByIndex(self,indices)
        % hierated from  matlab.io.datastore.PartitionableByIndex
          newds = copy(self);
          newds.imds = copy(self.imds);
          newds.OrderedIndices = indices;
      end

    end
    
        
    methods (Hidden)

        function frac = progress(self)
          % hierated from matlab.io.Datastore
            if hasdata(self)
                frac = (self.CurrentMiniBatchIndex - 1) / self.TotalNumberOfMiniBatches;
            else
                frac = 1;
            end
        end

    end
    methods (Access = private) % copied from denoisingImageDatastore.m
        
        function reorderIndexList(self,imdsIndexList)
           % Reorder OrderedIndices to be consistent with a new ordering of
           % the underlying imds. That is, when shuffle is called, we only
           % want to reorder imds, we don't want to end up with a truly
           % random shuffling of all of the observations because that will
           % drastically degrade performance by creating a situation where
           % each image patch is from a different source image.
            
           observationToImdsIndex = floor(( self.OrderedIndices -1) / self.PatchesPerImage) + 1;
           newObservationMapping = zeros(size(observationToImdsIndex),'like',observationToImdsIndex);
           currentIdxPos = 1;
           for i = 1:length(imdsIndexList)
              idx = imdsIndexList(i);
              sortedIdx = find(observationToImdsIndex == idx);
              newObservationMapping(currentIdxPos:(currentIdxPos+length(sortedIdx)-1)) = sortedIdx;
              currentIdxPos = currentIdxPos+length(sortedIdx);
           end
           self.OrderedIndices = newObservationMapping;
        end
        
        function [X,Y] = getNoisyPatches(self, images, numPatches)
            totalPatches = sum(numPatches);
            
            actualPatchSize = [self.PatchSize totalPatches];
            
            X = cell(totalPatches,1);
            Y = cell(totalPatches,1);
            
            count = 1;
            for imIndex = 1:length(numPatches)
                
                im = images{imIndex};
                
                patchSizeCheck = size(im,1) >= actualPatchSize(1) && ...
                    size(im,2) >= actualPatchSize(2);
                
                if ~patchSizeCheck
                    [~,fn,fe] = fileparts(self.imds.Files{imIndex}); 
                    error(message('images:noiseImageDatastore:expectPatchSmallerThanImage', [fn fe]));
                end
                                
                if strcmp(self.ChannelFormat,'rgb')
                    im = convertGrayscaleToRGB(im);
                else
                    im = convertRGBToGrayscale(im);
                end
                
                im = im2single(im);
                im = self.applyAugmentationPipeline(im);
                imNumPatches = numPatches(imIndex);
                
                rowLocations = randi(max(size(im,1)-actualPatchSize(1),1), imNumPatches, 1);
                colLocations = randi(max(size(im,2)-actualPatchSize(2),1), imNumPatches, 1);
                
                for index = 1:imNumPatches
                  % add noise
                    patch = im(rowLocations(index):rowLocations(index)+actualPatchSize(1)-1,...
                        colLocations(index):colLocations(index)+actualPatchSize(2)-1, :);
                      
                    [noisyPatch,residualNoise] = self.generateNoise(patch);
                    
                    Y{count} = residualNoise;
                    X{count} = noisyPatch;
                    count = count + 1;
                end
            end
        end
        
    end
    
    methods(Abstract, Access = protected)
      [noisyPatch,residualNoise] = generateNoise(self,cleanPatch)
    end

    methods (Access = private) % copied from augmentedImageDatasotre.m
        
        function Xout = applyAugmentationPipelineToBatch(self,X)
            if iscell(X)
                Xout = cellfun(@(c) self.applyAugmentationPipeline(c),X,'UniformOutput',false);
            else
                batchSize = size(X,4);
                Xout = cell(batchSize,1);
                for obs = 1:batchSize
                    Xout{obs} = self.augmentData(X(:,:,:,obs));
                end
            end
        end
        
        function Xout = applyAugmentationPipeline(self,X)
            Xout = self.augmentData(X);
        end
        
        function miniBatchData = augmentData(self,miniBatchData)
            if ~strcmp(self.ImageAugmenter,'none')
                miniBatchData = self.ImageAugmenter.augment(miniBatchData);
            end
        end
        
    end
end


function B = validateImagedatastore(ds)

validateattributes(ds, {'matlab.io.datastore.ImageDatastore'}, ...
    {'nonempty','vector'}, mfilename, 'IMDS');
validateattributes(ds.Files, {'cell'}, {'nonempty'}, mfilename, 'IMDS');

B = true;

end

function options = parseInputs(varargin)

parser = inputParser;
parser.KeepUnmatched = true;
parser.addParameter('PatchesPerImage',512,@validatePatchesPerImage);
parser.addParameter('PatchSize',50,@validatePatchSize);
parser.addParameter('BackgroundExecution',false,@validateBackgroundExecution);
parser.addParameter('DispatchInBackground',false,@validateDispatchInBackground);
parser.addParameter('ChannelFormat','grayscale',@validateChannelFormat);
parser.addParameter('DataAugmentation','none',@validateAugmentation);

parser.parse(varargin{:});
options = manageDispatchInBackgroundNameValue(parser);

if options.DispatchInBackground
  error('Matlab:bug','Dispatch in background is not properly supported yet');
%   error('Matlab:bug','Dispatch in background may throw unexpected error calling cuDNN: CUDNN_STATUS_BAD_PARAM in Matlab R2018a');
end
validOptions = {'rgb','grayscale'};
options.ChannelFormat = validatestring(options.ChannelFormat,validOptions, ...
    mfilename,'ChannelFormat');

end

function B = validatePatchesPerImage(PatchesPerImage)

attributes = {'nonempty','real','scalar', ...
    'positive','integer','finite','nonsparse','nonnan','nonzero'};

validateattributes(PatchesPerImage,images.internal.iptnumerictypes, attributes,...
    mfilename,'PatchesPerImage');

B = true;

end

function B = validatePatchSize(PatchSize)

attributes = {'nonempty','real','vector', ...
    'positive','integer','finite','nonsparse','nonnan','nonzero'};

validateattributes(PatchSize,images.internal.iptnumerictypes, attributes,...
    mfilename,'PatchSize');

if numel(PatchSize) > 2
    error(message('images:noiseImageDatastore:invalidPatchSize'));
end

B = true;

end

function B = validateBackgroundExecution(BackgroundExecution)

attributes = {'nonempty','scalar', ...
    'finite','nonsparse','nonnan'};
validateattributes(BackgroundExecution,{'logical'}, attributes,...
    mfilename,'BackgroundExecution');

B = true;

end

function B = validateDispatchInBackground(BackgroundExecution)

attributes = {'nonempty','scalar', ...
    'finite','nonsparse','nonnan'};
validateattributes(BackgroundExecution,{'logical'}, attributes,...
    mfilename,'DispatchInBackground');

B = true;

end

function B = validateChannelFormat(ChannelFormat)

supportedClasses = {'char','string'};
attributes = {'nonempty'};
validateattributes(ChannelFormat,supportedClasses,attributes,mfilename, ...
    'ChannelFormat');

B = true;
end

function B = validateAugmentation(valIn)

if ischar(valIn) || isstring(valIn)
    B = string('none').contains(lower(valIn)); %#ok<STRQUOT>
elseif isa(valIn,'imageDataAugmenter') && isscalar(valIn)
    B = true;
else
    B = false;
end

end

function im = convertRGBToGrayscale(im)
if ndims(im) == 3
    im = rgb2gray(im);
end
end

function im = convertGrayscaleToRGB(im)
if size(im,3) == 1
    im = repmat(im,[1 1 3]);
end
end

function resultsStruct = manageDispatchInBackgroundNameValue(p)

  resultsStruct = p.Results;

  DispatchInBackgroundSpecified = ~any(strncmp('DispatchInBackground',p.UsingDefaults,length('DispatchInBackground')));
  BackgroundExecutionSpecified = ~any(strncmp('BackgroundExecution',p.UsingDefaults,length('BackgroundExecution')));

  % In R2017b, BackgroundExecution was name used to control
  % DispatchInBackground. Allow either to be specified.
  if BackgroundExecutionSpecified && ~DispatchInBackgroundSpecified
      resultsStruct.DispatchInBackground = resultsStruct.BackgroundExecution;
  end

end
