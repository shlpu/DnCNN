function Y = vl_nnloss(X,c,dzdy,varargin)

% --------------------------------------------------------------------
% % ssim loss by Fang Li,2018-04-04
% --------------------------------------------------------------------
if nargin <= 2 || isempty(dzdy)
    load ssimpara
    ws = size(gaussFilt,1);
    n1 = ws; n2 = ws; delta1 = 1; delta2 = 1;
    
    [m,n,p,q] = size(X);
    K1 = 0.01; K2 = 0.03; R = 255;
    C(1) = (K1*R)^2; C(2) = (K2*R)^2;
    
    
    for i = 1:p
        for j = 1:q
            A = X(:,:,i,j);
            Ref = c(:,:,i,j);
            
            %  Weighted-mean and weighted-variance computations
            mux = filter2(gaussFilt,A,'valid');
            muy = filter2(gaussFilt,Ref,'valid');
            muxy = mux.*muy;
            mux2 = mux.^2;
            muy2 = muy.^2;
            
            sigmax2 = filter2(gaussFilt,A.^2,'valid') - mux2;
            sigmay2 = filter2(gaussFilt,Ref.^2,'valid') - muy2;
            sigmaxy = filter2(gaussFilt,A.*Ref,'valid') - muxy;
            
            A1 = 2*mux.*muy+C(1);
            A2 = 2*sigmaxy+C(2);
            B1 = mux2+muy2+C(1);
            B2 = sigmax2+sigmay2+C(2);
            
            ssim_map(:,:,i,j) = A1.*A2./(B1.*B2);
            ssim_val = mean(mean(ssim_map,1),2);
            Y = 1-ssim_val;
        end
    end
    
else
    
    load ssimpara
    ws = size(gaussFilt,1);
    n1 = ws; n2 = ws; delta1 = 1; delta2 = 1;
    
    [m,n,p,q] = size(X);
    K1 = 0.01; K2 = 0.03; R = 255;
    C(1) = (K1*R)^2; C(2) = (K2*R)^2;
    
    
    for i = 1:p
        for j = 1:q
            A = X(:,:,i,j);
            Ref = c(:,:,i,j);
            
            %  Weighted-mean and weighted-variance computations
            mux = filter2(gaussFilt,A,'valid');
            muy = filter2(gaussFilt,Ref,'valid');
            muxy = mux.*muy;
            mux2 = mux.^2;
            muy2 = muy.^2;
            
            sigmax2 = filter2(gaussFilt,A.^2,'valid') - mux2;
            sigmay2 = filter2(gaussFilt,Ref.^2,'valid') - muy2;
            sigmaxy = filter2(gaussFilt,A.*Ref,'valid') - muxy;
            
            A1 = 2*mux.*muy+C(1);
            A2 = 2*sigmaxy+C(2);
            B1 = mux2+muy2+C(1);
            B2 = sigmax2+sigmay2+C(2);
            
            ssim_map(:,:,i,j) = A1.*A2./(B1.*B2);
            ssim_val = mean(mean(ssim_map,1),2);
            
            A_patches = image2patches_fast(A, n1, n2, delta1, delta2);
            Ref_patches = image2patches_fast(Ref, n1, n2, delta1, delta2);
            gradientxSSIMxy = 0*A_patches;
            
            for k = 1:length(A_patches)
                x = A_patches(:,k);
                y = Ref_patches(:,k);
                factor = 2./(ws.^2.*B1(k).^2.*B2(k).^2);
                gradientxSSIMxy(:,k) = factor.*(A1(k).*B1(k).*(B2(k).*y-A2(k).*x)+B1(k).*B2(k).*(A2(k)-A1(k)).*muy(k)+A1(k).*A2(k).*(B1(k)-B2(k)).*mux(k));
            end
            
            gradientXSSIMXY(:,:,i,j) =  patches2image_fast(gradientxSSIMxy, m,n, delta1, delta2);
            
        end
    end
    Y = -1000*gradientXSSIMXY.*dzdy;% size(Y);
    
end
