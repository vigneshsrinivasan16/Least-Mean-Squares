L = 100; %Filter order
w_star = ones(L,1); %Optimal filter
n = 5000; %No of data points
initial_point = zeros(L,1); %w_0
mu = 0.001; %Step size, should be less than 0.019
K_empirical = 500;  %For ensemble average 
gamma_std_distribution = 3;

mag_E_stack = []; %For ensemble squared magnitude error

Rx = 0;

for j=1:K_empirical %Approximation to Expected Squared Error
    X = randn([L,n]); %Observed signal
    eta_n = randn([1 n]); %Random Noise
    d = w_star'*X + eta_n; %Desired Output
    [w_1,mag_E,~,d_hat] = LMS_alg(mu,d,initial_point,X,n);
    mag_E_stack = [mag_E_stack;mag_E];
end

mag_E_empirical = (1/K_empirical)*sum(mag_E_stack,1); 

itr = 1:1:n;

figure
plot(itr,mag_E_empirical)
xlabel("Iterations")
ylabel("Empirical Squared Error")


function [w_1,mag_E,E,d_hat] = LMS_alg(mu,d,initial_point,X,n)
E = []; 
mag_E = []; %Vector of squared magnitude errors at each time-step
w_1 = initial_point; %Filter Coefficient Initialization
d_hat = []; %Vector of predicted outputs at each time-step
for i=1:n
    y_1 = w_1'*X(:,i); %predicted output
    d_hat(i) = y_1; 
    e = d(i) - y_1; %error at ith time step
    E(i) = e;
    mag_E(i) = e^2;
    w_1 = w_1+mu*e*X(:,i); %gradient update
end 
end




