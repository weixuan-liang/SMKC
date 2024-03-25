function data =  pre_process(data)

[num,d] = size(data);
mean_row = mean(data);
std_row = std(data);
mean_total = repmat(mean_row, num, 1);
std_row = repmat(std_row, num, 1);

data = (data - mean_total) ./ std_row;
data(isnan(data)==1) = 0;
data = data ./ repmat(sqrt(sum(data.^2, 2)), 1,d);
end