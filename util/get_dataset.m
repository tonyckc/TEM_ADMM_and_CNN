% The theory formulation of TEM signal: $$S = \frac{C}{\tau}exp(-\frac{k^{2}}{\tau}t)+ n$$
% We implicitly define  $$\frac{C}{\tau}=k1$$ and $$\frac{k^{2}}{\tau}=k2$$,$$n$$ is explicitly defined as k3.
%% These values can be predefined according to your specific question.  
k1_max = 12e4;
k1_min = 5e4;
k1_num = 100;

k2_max = 0.4;
k2_min = 9.4;
k2_num =100;

k3_max = 15e2;
k3_min = 2e3;
k3_num = 10;
time_points = linspace(0,9,900); % the extracted time-points of received signal
sigma = 175; % the level of noise
% linespace is defined as the foundation of augmenting dataset. 
k1_linespace = linspace(k1_min,k1_max,k1_num);
k2_linespace = linspace(k2_min,k2_max,k2_num);
k3_linespace = linspace(k3_min,k3_max,k3_num);
clean_sig = [];
noise_sig = [];
num = 1;
%cshuffle = randperm(length(k1_num*k2_num*k3_num));
%% Simulate theory signal and Add Additive White Gaussian Noise. 
for k1_ite=1:k1_num
    for k2_ite=1:k2_num
        for k3_ite=1:k3_num
            tmp = k1_linespace(k1_ite)*exp(-k2_linespace(k2_ite)*time_points) + k3_linespace(k3_ite);
            noise_sig(num,:) = tmp + sigma* randn(size(tmp));
            clean_sig(num,:) = tmp;
            num = num + 1;
            disp(num);
            if num > 100000
                break;
            end
        end
    end
end
save('.\clean_signal.mat','clean_sig');
save('.\noise_signal.mat','noise_sig');


