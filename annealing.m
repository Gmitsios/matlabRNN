function out = annealing(start_data,end_data,num)

step = (end_data - start_data)/(num-1);
out = start_data:step:end_data;