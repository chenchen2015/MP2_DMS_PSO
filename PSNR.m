%% PSNR Calculator
function out_PSNR = PSNR(f1, f2)
% Calculate PSNR of the input image
k = 8;
% k denote for the bit-width of the image
fmax = 2.^k - 1;
a = fmax .^ 2;
e = double(f1) - double(f2);
[m, n] = size(e);
b = sum(sum(e.^2));
out_PSNR = 10 * log10(m * n * a / b);
end