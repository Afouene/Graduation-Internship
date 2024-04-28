Node = [3, 5, 7, 10];
Avg_Age_RL = [8.25, 13.9, 14.43, 18.76];
Avg_Age_RW = [12.1, 17.34, 18.97, 22.87];
Avg_Age_RR = [10.93, 15.23, 15.81, 19.02];

plot(Node, Avg_Age_RL, 'o-', 'DisplayName', 'Avg Age RL', 'LineWidth', 2);
hold on;

plot(Node, Avg_Age_RW, 's--', 'DisplayName', 'Avg Age RW', 'LineWidth', 2);

plot(Node, Avg_Age_RR, 'd-.', 'DisplayName', 'Avg Age RR', 'LineWidth', 2);

xlabel('Node');
ylabel('Average Age');
title('Average Age vs Node');
legend('show');