function result = cmc(dist, k)
    result = zeros(1, k);
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