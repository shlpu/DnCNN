width = 51;
sigma = 5;
img = randi(255,51,51)/255;

x = -(width/2):(width/2)-1;
w = exp(-1 * x.^2)/(2 * sigma^2);
w = w .* w';
w = w/sum(w(:));
% imshow(w,[])


mux = w*(img.^2);
mux = sum(mux(:))/numel(mux)
mean(img(:))
