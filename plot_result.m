function [ output_args ] = plot_result( s1, s2 )

figure(1);
plot(s1(:,1), s1(:,2));
hold on
plot(s1(:,1), s2(:,2), 'r');


end

