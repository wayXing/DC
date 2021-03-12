function [z,model] = pcaInit_v03(y,r)
% v_02: z = u * s
% v_03: z = u * sqrt(n-1), the Standardized scores.  z~normal 
% mypca
    [n,dim] = size(y);

    y_mean = mean(y);
    y = y - repmat(y_mean, n,1);
    
    if r >= 1   %calculated first r bases.
        [u,s,v] = svds(y,r);
        rank = r;
        eigenValue = diag(s);
        CumuEnergy = cumsum(eigenValue)./sum(eigenValue);
    elseif r<1
        [u,s,v] = svd(y,'econ');
        
        eigenValue = diag(s);
        CumuEnergy = cumsum(eigenValue)./sum(eigenValue);
        idx = find(CumuEnergy >=r);
        rank = idx(1);
        
        u = u(:,1:rank);
        s = s(1:rank,1:rank);
        v = v(:,1:rank);
    end
    
%     z = u;
%     z = u * s;
%     base = s*v';
    z = u * sqrt(n-1);
    v = v / sqrt(n-1);
    

    model.rank = rank;
    model.energy = CumuEnergy;
    model.eigenValue = eigenValue;
    model.s = s;
    model.v = v;
    model.y_mean = y_mean;
    
%     model.project = @(yte) (yte - repmat(y_mean, size(yte,1) ,1))*v*inv(s);
    invs = diag( (diag(s)+eps).^(-1) );
    model.project = @(yte) (yte - repmat(y_mean, size(yte,1) ,1))*v*invs;
%     model.project = @(yte) (yte - repmat(y_mean, size(yte,1) ,1))*v;
    
    model.recover = @(zte) zte*s*v' + repmat(y_mean, size(zte,1) ,1);
%     model.recover = @(zte) zte*v' + repmat(y_mean, size(zte,1) ,1);

end

