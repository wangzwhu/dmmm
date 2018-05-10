function [ result, dist, dist2 ] = calcMCMC2( M2, M, data2, data, idxa, idxb, idxtest )

dist = sqdist(data(:,idxa(idxtest)), data(:,idxb(idxtest)),M);

data2 = [data2 data];
dist2 = sqdist(data2(:,idxa(idxtest)), data2(:,idxb(idxtest)+1264),M2);

dist = dist - dist2;

result = zeros(1,size(dist,2));
for pairCounter=1:size(dist,2)
    distPair = dist(pairCounter,:);  
    [tmp,idx] = sort(distPair,'ascend');
    result(idx==pairCounter) = result(idx==pairCounter) + 1;
end

tmp = 0;
for counter=1:length(result)
    result(counter) = result(counter) + tmp;
    tmp = result(counter);
end

end

