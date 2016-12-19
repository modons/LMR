function Y = UK37_forward_model_func(filedir,X)

  
  % load posteriors for B coefficients and tau^2 variance
  coeff_file = strcat(filedir,'/','bayes_posterior_UK37.mat');
  load(coeff_file);
  
  % set up a Y matrix
  Y=NaN(size(X,1),length(tau2_draws_final));

  % set up spline parameters with set knots
  order=3; % spline order
  kn=augknt([-1 15 20 23 25 30],order); % knots

  % now set up a loop to draw B, tau, calculate spline, then draw from the joint posterior
  for i=1:length(tau2_draws_final)
    beta_now=b_draws_final(i,:);
    tau_now=tau2_draws_final(i);
    % spmak assembles the b-spline with the given knots and current coefficients
    bs_b=spmak(kn,beta_now);
    
    % fnxtr linearly extrapolates the spline to evaluate values at SSTs
    % outside the calibration range (0-30). w/o this, B-spline will return a NaN
    % at SSTs out of this range.
    bs=fnxtr(bs_b);
    
    % evaluate the mean value of the spline for your SST obs:
    mean_now=fnval(bs,X);
    
    % draw a sample from the distribution:
    Y(:,i)=normrnd(mean_now,sqrt(tau_now));
        
  end
    
end
